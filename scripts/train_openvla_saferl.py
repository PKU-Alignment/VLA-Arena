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
Upgraded OpenVLA SafeRL trainer:
- PPO-Lagrangian (single task)
- action-repeat acceleration
- rollout batch + minibatch PPO updates
"""

import argparse
import json
import random
import shutil
import time
from contextlib import nullcontext
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
    max_decision_steps: int = 180
    action_repeat: int = 2

    # SafeRL core
    algo: str = "ppo_lagrangian"
    num_episodes: int = 100
    max_updates: int = -1
    policy_lr: float = 1e-5

    # PPO
    rollout_episodes_per_update: int = 4
    update_epochs: int = 4
    minibatch_size: int = 64
    grad_accum_steps: int = 1
    ppo_clip_ratio: float = 0.2
    target_kl: float = 0.03
    entropy_coef: float = 0.01
    kl_ref_coef: float = 0.02
    use_generation_scores_for_old_logprob: bool = True

    # Returns / advantage
    gamma: float = 0.99
    normalize_advantage: bool = True
    adv_eps: float = 1e-8

    # Lagrangian
    init_lambda: float = 0.0
    lambda_lr: float = 0.01
    lambda_max: float = 100.0
    lambda_update_every: str = "update"  # "update" or "episode"
    cost_limit: float = 10.0

    # Sampling
    temperature: float = 1.0
    top_p: float = 1.0

    # Runtime/logging
    seed: int = 7
    max_grad_norm: float = 1.0
    log_every_updates: int = 1
    save_every_episodes: int = 10
    run_root_dir: str = "runs/saferl_openvla"
    use_wandb: bool = False
    wandb_project: str = "openvla-saferl"
    wandb_entity: str = "your_wandb_entity"
    merge_lora_for_eval: bool = True


@dataclass
class DecisionTransition:
    prompt: str
    image: np.ndarray
    action_token_ids: list[int]
    old_logprob: float
    reward: float
    cost: float


@dataclass
class EpisodeRollout:
    transitions: list[DecisionTransition]
    episode_reward: float
    episode_cost: float
    success: bool
    env_steps: int
    decisions: int
    duration_sec: float


@dataclass
class RolloutSample:
    prompt: str
    image: np.ndarray
    action_token_ids: list[int]
    old_logprob: float
    reward: float
    cost: float
    utility_return: float
    advantage: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_config(cfg: SafeRLConfig) -> None:
    if cfg.algo != "ppo_lagrangian":
        raise ValueError(f"Unsupported algo: {cfg.algo}")
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("Cannot use both 8-bit and 4-bit quantization.")
    if (cfg.load_in_8bit or cfg.load_in_4bit) and not cfg.use_lora:
        raise ValueError("Quantized training is supported only with LoRA in this script.")
    if cfg.rollout_episodes_per_update <= 0:
        raise ValueError("rollout_episodes_per_update must be > 0")
    if cfg.update_epochs <= 0:
        raise ValueError("update_epochs must be > 0")
    if cfg.minibatch_size <= 0:
        raise ValueError("minibatch_size must be > 0")
    if cfg.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")
    if cfg.action_repeat <= 0:
        raise ValueError("action_repeat must be > 0")
    if cfg.max_env_steps <= 0:
        raise ValueError("max_env_steps must be > 0")
    if cfg.max_decision_steps <= 0:
        raise ValueError("max_decision_steps must be > 0")
    if cfg.lambda_update_every not in {"update", "episode"}:
        raise ValueError("lambda_update_every must be one of: update, episode")


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


def load_norm_stats_if_available(
    model: OpenVLAForActionPrediction, checkpoint_dir: str | Path
) -> None:
    stats_path = Path(checkpoint_dir) / "dataset_statistics.json"
    if stats_path.is_file():
        with stats_path.open("r", encoding="utf-8") as f:
            norm_stats = json.load(f)
        model.norm_stats = norm_stats
        model.config.norm_stats = norm_stats


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


def compute_sequence_logprob_from_scores(
    scores: list[torch.Tensor], sampled_token_ids: list[int]
) -> float:
    logprob = 0.0
    for i, token_id in enumerate(sampled_token_ids):
        step_scores = scores[i].float()
        step_logprob = torch.log_softmax(step_scores, dim=-1)[0, token_id]
        logprob += float(step_logprob.item())
    return logprob


def compute_action_logprob_entropy_single(
    model: torch.nn.Module,
    processor: Any,
    prompt: str,
    image_array: np.ndarray,
    action_token_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=inputs["pixel_values"],
        )

    logits = outputs.logits.float()
    next_token_logits = logits[:, :-1, :]

    start = prompt_input_ids.shape[1] - 1
    end = start + action_ids.shape[1]
    action_logits = next_token_logits[:, start:end, :]

    action_log_probs = torch.log_softmax(action_logits, dim=-1)
    selected_log_probs = torch.gather(
        action_log_probs, dim=-1, index=action_ids.unsqueeze(-1)
    ).squeeze(-1)
    sequence_logprob = selected_log_probs.sum(dim=1).squeeze(0)

    action_probs = torch.softmax(action_logits, dim=-1)
    token_entropy = -(action_probs * action_log_probs).sum(dim=-1)
    sequence_entropy = token_entropy.mean(dim=1).squeeze(0)

    return sequence_logprob, sequence_entropy


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
    use_generation_scores_for_old_logprob: bool,
) -> tuple[np.ndarray, list[int], float]:
    image = Image.fromarray(image_array).convert("RGB")
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = input_ids.clone()
        input_ids[:, -1] = 29871

    action_dim = base_vla.get_action_dim(unnorm_key)
    with torch.no_grad():
        generated = model.generate(
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
            return_dict_in_generate=True,
            output_scores=use_generation_scores_for_old_logprob,
        )

    sampled_token_ids = (
        generated.sequences[0, -action_dim:].detach().cpu().numpy().astype(int).tolist()
    )
    action = decode_action_from_tokens(
        base_vla, np.asarray(sampled_token_ids, dtype=np.int64), unnorm_key
    )

    if use_generation_scores_for_old_logprob and generated.scores is not None:
        old_logprob = compute_sequence_logprob_from_scores(
            list(generated.scores), sampled_token_ids
        )
    else:
        with torch.no_grad():
            old_lp_t, _ = compute_action_logprob_entropy_single(
                model=model,
                processor=processor,
                prompt=prompt,
                image_array=image_array,
                action_token_ids=sampled_token_ids,
                device=device,
            )
        old_logprob = float(old_lp_t.item())

    return action, sampled_token_ids, old_logprob


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
    update_idx: int,
    lagrangian_lambda: float,
) -> None:
    checkpoint_dir = run_dir / f"episode_{episode_idx:06d}_update_{update_idx:06d}"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state_payload = {
        "episode": episode_idx,
        "update": update_idx,
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


def collect_rollout_batch(
    cfg: SafeRLConfig,
    model: torch.nn.Module,
    processor: Any,
    base_vla: OpenVLAForActionPrediction,
    unnorm_key: str,
    env: Any,
    task_description: str,
    initial_states: Any,
    rng: np.random.Generator,
    num_episodes_to_collect: int,
    device: torch.device,
) -> list[EpisodeRollout]:
    episodes: list[EpisodeRollout] = []

    for _ in range(num_episodes_to_collect):
        ep_start = time.time()

        env.reset()
        if initial_states is not None and len(initial_states) > 0:
            state_idx = int(rng.integers(0, len(initial_states)))
            obs = env.set_init_state(initial_states[state_idx])
        else:
            obs = env.get_observation()

        transitions: list[DecisionTransition] = []
        episode_reward = 0.0
        episode_cost = 0.0
        success = False
        env_steps = 0
        decisions = 0

        for _ in range(cfg.num_steps_wait):
            obs, _, done, _ = env.step(get_vla_arena_dummy_action("openvla"))
            env_steps += 1
            if done:
                success = True
                break

        prompt = build_prompt(cfg.pretrained_checkpoint, task_description)

        while (
            not success
            and decisions < cfg.max_decision_steps
            and env_steps < cfg.max_env_steps + cfg.num_steps_wait
        ):
            policy_image = preprocess_policy_image(obs)
            action, sampled_token_ids, old_logprob = sample_action_tokens(
                model=model,
                processor=processor,
                base_vla=base_vla,
                prompt=prompt,
                image_array=policy_image,
                unnorm_key=unnorm_key,
                device=device,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                use_generation_scores_for_old_logprob=cfg.use_generation_scores_for_old_logprob,
            )

            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

            decision_reward = 0.0
            decision_cost = 0.0
            for _ in range(cfg.action_repeat):
                obs, reward, done, info = env.step(action.tolist())
                env_steps += 1
                decision_reward += float(reward)
                decision_cost += float(info.get("cost", 0.0))

                if done:
                    success = True
                    break
                if env_steps >= cfg.max_env_steps + cfg.num_steps_wait:
                    break

            decisions += 1
            episode_reward += decision_reward
            episode_cost += decision_cost
            transitions.append(
                DecisionTransition(
                    prompt=prompt,
                    image=policy_image.copy(),
                    action_token_ids=sampled_token_ids,
                    old_logprob=old_logprob,
                    reward=decision_reward,
                    cost=decision_cost,
                )
            )

        episodes.append(
            EpisodeRollout(
                transitions=transitions,
                episode_reward=episode_reward,
                episode_cost=episode_cost,
                success=success,
                env_steps=env_steps,
                decisions=decisions,
                duration_sec=time.time() - ep_start,
            )
        )

    return episodes


def build_rollout_samples(
    episodes: list[EpisodeRollout],
    gamma: float,
    lagrangian_lambda: float,
    normalize_advantage: bool,
    adv_eps: float,
) -> list[RolloutSample]:
    samples: list[RolloutSample] = []

    for ep in episodes:
        utilities = [
            tr.reward - lagrangian_lambda * tr.cost for tr in ep.transitions
        ]
        returns = compute_discounted_returns(utilities, gamma)

        for tr, ret in zip(ep.transitions, returns):
            samples.append(
                RolloutSample(
                    prompt=tr.prompt,
                    image=tr.image,
                    action_token_ids=tr.action_token_ids,
                    old_logprob=tr.old_logprob,
                    reward=tr.reward,
                    cost=tr.cost,
                    utility_return=ret,
                )
            )

    if not samples:
        return samples

    returns_np = np.asarray([s.utility_return for s in samples], dtype=np.float32)
    baseline = float(returns_np.mean())
    advantages = returns_np - baseline

    if normalize_advantage:
        adv_std = float(advantages.std())
        advantages = (advantages - float(advantages.mean())) / max(adv_std, adv_eps)

    for s, a in zip(samples, advantages):
        s.advantage = float(a)

    return samples


def maybe_disable_adapter_ctx(model: torch.nn.Module):
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


def count_nonzero_grad_params(model: torch.nn.Module) -> int:
    count = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            if torch.count_nonzero(p.grad).item() > 0:
                count += 1
    return count


def ppo_update(
    cfg: SafeRLConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    processor: Any,
    samples: list[RolloutSample],
    device: torch.device,
) -> dict[str, float]:
    if not samples:
        return {
            "ppo_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "approx_kl_ref": 0.0,
            "clip_fraction": 0.0,
            "adv_mean": 0.0,
            "adv_std": 0.0,
            "num_samples": 0.0,
            "nonzero_grad_params": 0.0,
            "stopped_early_kl": 0.0,
        }

    model.train()
    indices = np.arange(len(samples))

    total_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_approx_kl_ref = 0.0
    total_clip_fraction = 0.0
    total_mb = 0
    stopped_early_kl = 0

    optimizer.zero_grad(set_to_none=True)
    accum_counter = 0
    max_nonzero_grad_params = 0

    for epoch_idx in range(cfg.update_epochs):
        np.random.shuffle(indices)

        for mb_start in range(0, len(indices), cfg.minibatch_size):
            mb_idx = indices[mb_start : mb_start + cfg.minibatch_size]
            mb_samples = [samples[i] for i in mb_idx]

            current_logprobs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            old_logprobs: list[float] = []
            advantages: list[float] = []

            for sample in mb_samples:
                cur_lp, ent = compute_action_logprob_entropy_single(
                    model=model,
                    processor=processor,
                    prompt=sample.prompt,
                    image_array=sample.image,
                    action_token_ids=sample.action_token_ids,
                    device=device,
                )
                current_logprobs.append(cur_lp)
                entropies.append(ent)
                old_logprobs.append(sample.old_logprob)
                advantages.append(sample.advantage)

            current_lp_t = torch.stack(current_logprobs)
            entropy_t = torch.stack(entropies)
            old_lp_t = torch.tensor(old_logprobs, device=device, dtype=torch.float32)
            adv_t = torch.tensor(advantages, device=device, dtype=torch.float32)

            ratio = torch.exp(current_lp_t - old_lp_t)
            unclipped_obj = ratio * adv_t
            clipped_obj = torch.clamp(
                ratio,
                1.0 - cfg.ppo_clip_ratio,
                1.0 + cfg.ppo_clip_ratio,
            ) * adv_t
            ppo_loss = -torch.mean(torch.minimum(unclipped_obj, clipped_obj))

            entropy_mean = entropy_t.mean()
            approx_kl = torch.mean(old_lp_t - current_lp_t)
            clip_fraction = (
                (torch.abs(ratio - 1.0) > cfg.ppo_clip_ratio).float().mean()
            )

            approx_kl_ref = torch.tensor(0.0, device=device)
            if cfg.kl_ref_coef > 0.0 and cfg.use_lora:
                with torch.no_grad():
                    with maybe_disable_adapter_ctx(model):
                        ref_logprobs: list[torch.Tensor] = []
                        for sample in mb_samples:
                            ref_lp, _ = compute_action_logprob_entropy_single(
                                model=model,
                                processor=processor,
                                prompt=sample.prompt,
                                image_array=sample.image,
                                action_token_ids=sample.action_token_ids,
                                device=device,
                            )
                            ref_logprobs.append(ref_lp)
                ref_lp_t = torch.stack(ref_logprobs).detach()
                approx_kl_ref = torch.mean(current_lp_t - ref_lp_t)

            loss = ppo_loss - cfg.entropy_coef * entropy_mean + cfg.kl_ref_coef * approx_kl_ref
            (loss / cfg.grad_accum_steps).backward()

            accum_counter += 1
            if accum_counter % cfg.grad_accum_steps == 0:
                max_nonzero_grad_params = max(
                    max_nonzero_grad_params,
                    count_nonzero_grad_params(model),
                )
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=cfg.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(ppo_loss.detach().cpu().item())
            total_entropy += float(entropy_mean.detach().cpu().item())
            total_approx_kl += float(approx_kl.detach().cpu().item())
            total_approx_kl_ref += float(approx_kl_ref.detach().cpu().item())
            total_clip_fraction += float(clip_fraction.detach().cpu().item())
            total_mb += 1

            if cfg.target_kl > 0 and float(approx_kl.detach().cpu().item()) > cfg.target_kl:
                stopped_early_kl = 1
                break

        if stopped_early_kl:
            break

    if accum_counter % cfg.grad_accum_steps != 0:
        max_nonzero_grad_params = max(
            max_nonzero_grad_params,
            count_nonzero_grad_params(model),
        )
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=cfg.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    adv_np = np.asarray([s.advantage for s in samples], dtype=np.float32)

    denom = max(total_mb, 1)
    return {
        "ppo_loss": total_loss / denom,
        "entropy": total_entropy / denom,
        "approx_kl": total_approx_kl / denom,
        "approx_kl_ref": total_approx_kl_ref / denom,
        "clip_fraction": total_clip_fraction / denom,
        "adv_mean": float(adv_np.mean()) if len(adv_np) > 0 else 0.0,
        "adv_std": float(adv_np.std()) if len(adv_np) > 0 else 0.0,
        "num_samples": float(len(samples)),
        "nonzero_grad_params": float(max_nonzero_grad_params),
        "stopped_early_kl": float(stopped_early_kl),
    }


def update_lagrangian_lambda(
    cfg: SafeRLConfig,
    lagrangian_lambda: float,
    episodes: list[EpisodeRollout],
) -> float:
    if not episodes:
        return lagrangian_lambda

    if cfg.lambda_update_every == "episode":
        for ep in episodes:
            lagrangian_lambda = lagrangian_lambda + cfg.lambda_lr * (
                ep.episode_cost - cfg.cost_limit
            )
            lagrangian_lambda = max(0.0, min(cfg.lambda_max, lagrangian_lambda))
        return lagrangian_lambda

    mean_cost = float(np.mean([ep.episode_cost for ep in episodes]))
    lagrangian_lambda = lagrangian_lambda + cfg.lambda_lr * (
        mean_cost - cfg.cost_limit
    )
    lagrangian_lambda = max(0.0, min(cfg.lambda_max, lagrangian_lambda))
    return lagrangian_lambda


def main(config: SafeRLConfig | str | Path) -> None:
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        try:
            cfg = draccus.parse(SafeRLConfig, config_path=str(config_path), args=[])
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse config {config_path}: {e}"
            ) from e
    elif isinstance(config, SafeRLConfig):
        cfg = config
    else:
        raise ValueError(
            f"Unsupported config type: {type(config)}. Expected SafeRLConfig or path string."
        )

    validate_config(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires CUDA for practical execution.")

    device = torch.device("cuda:0")
    set_seed(cfg.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"openvla-saferl-ppo-{cfg.task_suite_name}-L{cfg.task_level}-T{cfg.task_id}-{timestamp}"
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
    global_episode = 0
    update_idx = 0
    total_successes = 0
    next_save_episode = cfg.save_every_episodes

    try:
        while global_episode < cfg.num_episodes:
            if cfg.max_updates >= 0 and update_idx >= cfg.max_updates:
                break

            episodes_this_rollout = min(
                cfg.rollout_episodes_per_update,
                cfg.num_episodes - global_episode,
            )
            rollout_start = time.time()
            episodes = collect_rollout_batch(
                cfg=cfg,
                model=model,
                processor=processor,
                base_vla=base_vla,
                unnorm_key=unnorm_key,
                env=env,
                task_description=task_description,
                initial_states=initial_states,
                rng=rng,
                num_episodes_to_collect=episodes_this_rollout,
                device=device,
            )
            rollout_time = time.time() - rollout_start

            global_episode += len(episodes)
            update_idx += 1
            total_successes += sum(1 for ep in episodes if ep.success)

            samples = build_rollout_samples(
                episodes=episodes,
                gamma=cfg.gamma,
                lagrangian_lambda=lagrangian_lambda,
                normalize_advantage=cfg.normalize_advantage,
                adv_eps=cfg.adv_eps,
            )

            update_metrics = ppo_update(
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                processor=processor,
                samples=samples,
                device=device,
            )

            lagrangian_lambda = update_lagrangian_lambda(
                cfg=cfg,
                lagrangian_lambda=lagrangian_lambda,
                episodes=episodes,
            )

            total_decisions = sum(ep.decisions for ep in episodes)
            total_env_steps = sum(ep.env_steps for ep in episodes)
            total_reward = sum(ep.episode_reward for ep in episodes)
            total_cost = sum(ep.episode_cost for ep in episodes)
            success_rate = total_successes / float(global_episode)

            metrics = {
                "update": update_idx,
                "episode": global_episode,
                "episodes_in_update": len(episodes),
                "episode_reward_mean": total_reward / max(len(episodes), 1),
                "episode_cost_mean": total_cost / max(len(episodes), 1),
                "lambda": lagrangian_lambda,
                "success_rate": success_rate,
                "num_decisions": total_decisions,
                "num_env_steps": total_env_steps,
                "decisions_per_sec": total_decisions / max(rollout_time, 1e-6),
                "env_steps_per_sec": total_env_steps / max(rollout_time, 1e-6),
                "rollout_time_sec": rollout_time,
                **update_metrics,
            }

            with metrics_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

            if wandb_run is not None:
                wandb_run.log(metrics, step=update_idx)

            if update_idx % cfg.log_every_updates == 0:
                print(
                    f"[Update {update_idx:04d}] "
                    f"episodes={global_episode} "
                    f"reward_mean={metrics['episode_reward_mean']:.4f} "
                    f"cost_mean={metrics['episode_cost_mean']:.4f} "
                    f"lambda={lagrangian_lambda:.4f} "
                    f"ppo_loss={metrics['ppo_loss']:.6f} "
                    f"kl={metrics['approx_kl']:.6f} "
                    f"clip_frac={metrics['clip_fraction']:.4f} "
                    f"succ_rate={success_rate:.4f} "
                    f"dec/s={metrics['decisions_per_sec']:.2f}"
                )

            while cfg.save_every_episodes > 0 and global_episode >= next_save_episode:
                save_checkpoint(
                    cfg=cfg,
                    model=model,
                    processor=processor,
                    base_vla=base_vla,
                    run_dir=run_dir,
                    episode_idx=global_episode,
                    update_idx=update_idx,
                    lagrangian_lambda=lagrangian_lambda,
                )
                next_save_episode += cfg.save_every_episodes

        save_checkpoint(
            cfg=cfg,
            model=model,
            processor=processor,
            base_vla=base_vla,
            run_dir=run_dir,
            episode_idx=global_episode,
            update_idx=update_idx,
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
        description="Upgraded OpenVLA SafeRL trainer (PPO-Lagrangian)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the yaml config file.",
    )
    args = parser.parse_args()
    main(args.config)
