# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Minimal SafeRL training script for OpenVLA using REINFORCE + Lagrangian.

This script is intentionally simple and single-GPU:
- single task from a safety suite
- online rollout in VLA-Arena env
- cost from env info["cost"]
- policy gradient update over sampled action tokens
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
import tqdm
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)

from vla_arena.models.openvla.experiments.robot.robot_utils import (
    invert_gripper_action,
    normalize_gripper_action,
)
from vla_arena.models.openvla.experiments.robot.vla_arena.vla_arena_utils import (
    get_vla_arena_dummy_action,
    get_vla_arena_env,
    get_vla_arena_image,
)
from vla_arena.models.openvla.prismatic.extern.hf.configuration_prismatic import (
    OpenVLAConfig,
)
from vla_arena.models.openvla.prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction,
)
from vla_arena.models.openvla.prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from vla_arena.vla_arena import benchmark


IGNORE_INDEX = -100
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


@dataclass
class SafeRLConfig:
    # Model
    pretrained_checkpoint: str = "/path/to/your/openvla_checkpoint"
    unnorm_key: str = "vla_arena"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

    # Environment/task
    task_suite_name: str = "safety_static_obstacles"
    task_level: int = 0
    task_id: int = 0
    num_steps_wait: int = 10
    max_env_steps: int = 300

    # SafeRL
    num_episodes: int = 100
    policy_lr: float = 1e-5
    gamma: float = 0.99
    init_lambda: float = 0.0
    lambda_lr: float = 0.01
    cost_limit: float = 10.0
    temperature: float = 1.0
    top_p: float = 1.0
    max_grad_norm: float = 1.0

    # Runtime/logging
    seed: int = 7
    save_every_episodes: int = 10
    run_root_dir: str = "runs/saferl_openvla"
    use_wandb: bool = False
    wandb_project: str = "openvla-saferl"
    wandb_entity: str = "your_wandb_entity"
    merge_lora_for_eval: bool = True


@dataclass
class RolloutStep:
    prompt: str
    image: np.ndarray
    action_token_ids: list[int]
    reward: float
    cost: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def register_openvla_hf_autoclasses() -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def get_base_vla(model: torch.nn.Module) -> OpenVLAForActionPrediction:
    if isinstance(model, PeftModel):
        base = model.get_base_model()
        if hasattr(base, "model"):
            base = base.model
        return base
    return model  # type: ignore[return-value]


def ensure_unnorm_key(cfg: SafeRLConfig, base_vla: OpenVLAForActionPrediction) -> str:
    key = cfg.unnorm_key
    if key not in base_vla.norm_stats and f"{key}_no_noops" in base_vla.norm_stats:
        key = f"{key}_no_noops"
    if key not in base_vla.norm_stats:
        raise ValueError(
            f"Action un-norm key {key} not found. Available keys: {list(base_vla.norm_stats.keys())}"
        )
    return key


def build_prompt(pretrained_checkpoint: str, task_label: str) -> str:
    task = task_label.lower()
    if "openvla-v01" in str(pretrained_checkpoint):
        return (
            f"{OPENVLA_V01_SYSTEM_PROMPT} "
            f"USER: What action should the robot take to {task}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task}?\nOut:"


def load_norm_stats_if_available(
    model: OpenVLAForActionPrediction, checkpoint_dir: str | Path
) -> None:
    stats_path = Path(checkpoint_dir) / "dataset_statistics.json"
    if stats_path.is_file():
        with stats_path.open("r", encoding="utf-8") as f:
            norm_stats = json.load(f)
        model.norm_stats = norm_stats
        model.config.norm_stats = norm_stats


def decode_action_from_tokens(
    base_vla: OpenVLAForActionPrediction, token_ids: np.ndarray, unnorm_key: str
) -> np.ndarray:
    discretized_actions = base_vla.vocab_size - token_ids
    discretized_actions = np.clip(
        discretized_actions - 1,
        a_min=0,
        a_max=base_vla.bin_centers.shape[0] - 1,
    )
    normalized_actions = base_vla.bin_centers[discretized_actions]

    action_norm_stats = base_vla.get_action_stats(unnorm_key)
    mask = action_norm_stats.get(
        "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
    )
    action_high = np.asarray(action_norm_stats["q99"])
    action_low = np.asarray(action_norm_stats["q01"])

    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions.astype(np.float32)


def preprocess_policy_image(obs: dict[str, Any]) -> np.ndarray:
    return get_vla_arena_image(obs, resize_size=224)


def sample_action_tokens(
    model: torch.nn.Module,
    processor: Any,
    base_vla: OpenVLAForActionPrediction,
    prompt: str,
    image_array: np.ndarray,
    unnorm_key: str,
    device: torch.device,
    temperature: float,
    top_p: float,
) -> tuple[np.ndarray, list[int]]:
    image = Image.fromarray(image_array).convert("RGB")
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = input_ids.clone()
        input_ids[:, -1] = 29871

    action_dim = base_vla.get_action_dim(unnorm_key)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            min_new_tokens=action_dim,
            max_new_tokens=action_dim,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    sampled_token_ids = generated_ids[0, -action_dim:].detach().cpu().numpy()
    action = decode_action_from_tokens(base_vla, sampled_token_ids, unnorm_key)
    return action, sampled_token_ids.astype(int).tolist()


def compute_action_nll(
    model: torch.nn.Module,
    processor: Any,
    prompt: str,
    image_array: np.ndarray,
    action_token_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    image = Image.fromarray(image_array).convert("RGB")
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    prompt_input_ids = inputs["input_ids"]
    if not torch.all(prompt_input_ids[:, -1] == 29871):
        prompt_input_ids = prompt_input_ids.clone()
        prompt_input_ids[:, -1] = 29871

    action_ids = torch.tensor(
        action_token_ids, dtype=prompt_input_ids.dtype, device=device
    ).unsqueeze(0)
    full_input_ids = torch.cat([prompt_input_ids, action_ids], dim=1)
    full_attention_mask = torch.ones_like(full_input_ids, device=device)

    labels = torch.full_like(full_input_ids, fill_value=IGNORE_INDEX)
    labels[:, -action_ids.shape[1] :] = full_input_ids[:, -action_ids.shape[1] :]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=inputs["pixel_values"],
            labels=labels,
        )

    return outputs.loss.float()


def compute_discounted_returns(values: list[float], gamma: float) -> list[float]:
    returns: list[float] = []
    running = 0.0
    for value in reversed(values):
        running = value + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_dataset_statistics(
    out_dir: Path, base_vla: OpenVLAForActionPrediction
) -> None:
    stats_path = out_dir / "dataset_statistics.json"
    stats_path.write_text(
        json.dumps(base_vla.norm_stats, indent=2),
        encoding="utf-8",
    )


def merge_and_save_lora_checkpoint(
    cfg: SafeRLConfig,
    processor: Any,
    adapter_dir: Path,
    merged_dir: Path,
) -> None:
    base_model = OpenVLAForActionPrediction.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = merged_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(merged_dir))
    merged_model.save_pretrained(str(merged_dir))
    save_dataset_statistics(merged_dir, merged_model)  # type: ignore[arg-type]


def save_checkpoint(
    cfg: SafeRLConfig,
    model: torch.nn.Module,
    processor: Any,
    base_vla: OpenVLAForActionPrediction,
    run_dir: Path,
    episode_idx: int,
    lagrangian_lambda: float,
) -> None:
    checkpoint_dir = run_dir / f"episode_{episode_idx:06d}"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state_payload = {
        "episode": episode_idx,
        "lambda": lagrangian_lambda,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(checkpoint_dir / "training_state.json", state_payload)

    if cfg.use_lora:
        adapter_dir = checkpoint_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        processor.save_pretrained(str(adapter_dir))
        save_dataset_statistics(adapter_dir, base_vla)

        if cfg.merge_lora_for_eval:
            merge_and_save_lora_checkpoint(
                cfg,
                processor=processor,
                adapter_dir=adapter_dir,
                merged_dir=checkpoint_dir / "merged",
            )
    else:
        processor.save_pretrained(str(checkpoint_dir))
        model.save_pretrained(str(checkpoint_dir))
        save_dataset_statistics(checkpoint_dir, base_vla)

    latest_dir = run_dir / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(checkpoint_dir, latest_dir)


def create_model_and_optimizer(
    cfg: SafeRLConfig, device: torch.device
) -> tuple[torch.nn.Module, Any, OpenVLAForActionPrediction, torch.optim.Optimizer]:
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("Cannot use both 8-bit and 4-bit quantization.")
    if (cfg.load_in_8bit or cfg.load_in_4bit) and not cfg.use_lora:
        raise ValueError("Quantized training is supported only with LoRA in this script.")

    register_openvla_hf_autoclasses()

    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(
        cfg.pretrained_checkpoint, trust_remote_code=True
    )
    model: torch.nn.Module = OpenVLAForActionPrediction.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        model = model.to(device)

    base_vla = get_base_vla(model)
    load_norm_stats_if_available(base_vla, cfg.pretrained_checkpoint)

    if cfg.use_lora:
        if cfg.load_in_8bit or cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
        for param in get_base_vla(model).parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters were found.")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.policy_lr, weight_decay=0.0
    )
    model.train()
    return model, processor, get_base_vla(model), optimizer


def main(config: SafeRLConfig | str | Path) -> None:
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        cfg = draccus.parse(SafeRLConfig, config_path=str(config_path), args=[])
    elif isinstance(config, SafeRLConfig):
        cfg = config
    else:
        raise ValueError(
            f"Unsupported config type: {type(config)}. Expected SafeRLConfig or path string."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires CUDA for practical execution.")

    device = torch.device("cuda:0")
    set_seed(cfg.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"openvla-saferl-{cfg.task_suite_name}-L{cfg.task_level}-T{cfg.task_id}-{timestamp}"
    )
    run_dir = Path(cfg.run_root_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "train_metrics.jsonl"
    save_json(run_dir / "resolved_config.json", asdict(cfg))

    model, processor, base_vla, optimizer = create_model_and_optimizer(cfg, device)
    unnorm_key = ensure_unnorm_key(cfg, base_vla)

    benchmark_dict = benchmark.get_benchmark_dict()
    if cfg.task_suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown task suite: {cfg.task_suite_name}. "
            f"Available options: {list(benchmark_dict.keys())}"
        )
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task_by_level_id(cfg.task_level, cfg.task_id)
    if task is None:
        raise ValueError(
            f"Task not found for suite={cfg.task_suite_name}, "
            f"level={cfg.task_level}, task_id={cfg.task_id}"
        )

    env, task_description = get_vla_arena_env(
        task,
        model_family="openvla",
        resolution=256,
        add_noise=False,
        randomize_color=False,
        adjust_light=False,
        camera_offset=False,
    )
    if isinstance(task_description, list):
        task_description = task_description[0]

    initial_states = task_suite.get_task_init_states(cfg.task_level, cfg.task_id)
    rng = np.random.default_rng(cfg.seed)

    wandb_run = None
    if cfg.use_wandb:
        import wandb

        wandb_run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_name,
            config=asdict(cfg),
        )

    lagrangian_lambda = float(cfg.init_lambda)
    success_count = 0

    try:
        for episode_idx in tqdm.tqdm(
            range(1, cfg.num_episodes + 1), desc="SafeRL episodes"
        ):
            env.reset()
            if initial_states is not None and len(initial_states) > 0:
                state_idx = int(rng.integers(0, len(initial_states)))
                obs = env.set_init_state(initial_states[state_idx])
            else:
                obs = env.get_observation()

            step_data: list[RolloutStep] = []
            episode_reward = 0.0
            episode_cost = 0.0
            success = False

            for t in range(cfg.max_env_steps + cfg.num_steps_wait):
                if t < cfg.num_steps_wait:
                    obs, _, done, _ = env.step(get_vla_arena_dummy_action("openvla"))
                    if done:
                        success = True
                        break
                    continue

                policy_image = preprocess_policy_image(obs)
                prompt = build_prompt(cfg.pretrained_checkpoint, task_description)
                action, sampled_token_ids = sample_action_tokens(
                    model=model,
                    processor=processor,
                    base_vla=base_vla,
                    prompt=prompt,
                    image_array=policy_image,
                    unnorm_key=unnorm_key,
                    device=device,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                )

                # Match evaluator behavior for gripper sign conventions.
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)

                obs, reward, done, info = env.step(action.tolist())
                step_cost = float(info.get("cost", 0.0))

                episode_reward += float(reward)
                episode_cost += step_cost
                step_data.append(
                    RolloutStep(
                        prompt=prompt,
                        image=policy_image.copy(),
                        action_token_ids=sampled_token_ids,
                        reward=float(reward),
                        cost=step_cost,
                    )
                )

                if done:
                    success = True
                    break

            if success:
                success_count += 1

            weighted_loss_sum = 0.0
            nonzero_grad_params = 0
            if step_data:
                # REINFORCE objective weights from discounted utility returns.
                utilities = [
                    step.reward - lagrangian_lambda * step.cost
                    for step in step_data
                ]
                returns = compute_discounted_returns(utilities, cfg.gamma)

                optimizer.zero_grad(set_to_none=True)
                for step_idx, step in enumerate(step_data):
                    nll = compute_action_nll(
                        model=model,
                        processor=processor,
                        prompt=step.prompt,
                        image_array=step.image,
                        action_token_ids=step.action_token_ids,
                        device=device,
                    )
                    weight = returns[step_idx] / len(step_data)
                    step_loss = nll * float(weight)
                    step_loss.backward()
                    weighted_loss_sum += float(step_loss.detach().cpu().item())

                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        if torch.count_nonzero(param.grad).item() > 0:
                            nonzero_grad_params += 1

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=cfg.max_grad_norm,
                )
                optimizer.step()

            lagrangian_lambda = max(
                0.0,
                lagrangian_lambda + cfg.lambda_lr * (episode_cost - cfg.cost_limit),
            )

            metrics = {
                "episode": episode_idx,
                "episode_reward": episode_reward,
                "episode_cost": episode_cost,
                "lambda": lagrangian_lambda,
                "policy_loss": weighted_loss_sum,
                "num_steps": len(step_data),
                "success": float(success),
                "success_rate": success_count / float(episode_idx),
                "nonzero_grad_params": nonzero_grad_params,
            }
            with metrics_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

            if wandb_run is not None:
                wandb_run.log(metrics, step=episode_idx)

            print(
                f"[Episode {episode_idx:04d}] reward={episode_reward:.4f} "
                f"cost={episode_cost:.4f} lambda={lagrangian_lambda:.4f} "
                f"loss={weighted_loss_sum:.6f} success={success} "
                f"nonzero_grad_params={nonzero_grad_params}"
            )

            if episode_idx % cfg.save_every_episodes == 0:
                save_checkpoint(
                    cfg=cfg,
                    model=model,
                    processor=processor,
                    base_vla=base_vla,
                    run_dir=run_dir,
                    episode_idx=episode_idx,
                    lagrangian_lambda=lagrangian_lambda,
                )

        # Final checkpoint
        save_checkpoint(
            cfg=cfg,
            model=model,
            processor=processor,
            base_vla=base_vla,
            run_dir=run_dir,
            episode_idx=cfg.num_episodes,
            lagrangian_lambda=lagrangian_lambda,
        )

    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()

    print(f"Training finished. Run directory: {run_dir}")
    print(f"Metrics jsonl: {metrics_file}")
    print(f"Latest checkpoint: {run_dir / 'latest'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal OpenVLA SafeRL trainer (REINFORCE + Lagrangian)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the yaml config file.",
    )
    args = parser.parse_args()
    main(args.config)
