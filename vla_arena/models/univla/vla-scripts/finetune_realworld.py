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

import os
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
import wandb
from accelerate import Accelerator, PartialState
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)

from vla_arena.models.univla.prismatic.extern.hf.configuration_prismatic import (
    OpenVLAConfig,
)
from vla_arena.models.univla.prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction,
)
from vla_arena.models.univla.prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from vla_arena.models.univla.prismatic.models.backbones.llm.prompting import (
    PurePromptBuilder,
)
from vla_arena.models.univla.prismatic.vla.action_tokenizer import (
    ActionTokenizer,
)
from vla_arena.models.univla.prismatic.vla.datasets.real_world_dataset import (
    find_all_hdf5,
    load_data_univla,
)


# Sane Defaults
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from vla_arena.models.univla.prismatic.models.policy.transformer_utils import (
    MAPBlock,
)


class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size=5, hidden_dim=512):
        super().__init__()
        self.attn_pool = MAPBlock(
            n_latents=1,
            vis_dim=4096,
            embed_dim=hidden_dim,
            n_heads=hidden_dim // 64,
        )
        self.visual_pool = MAPBlock(
            n_latents=1,
            vis_dim=4096,
            embed_dim=hidden_dim,
            n_heads=hidden_dim // 64,
        )
        self.proprio_proj = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, window_size * 7),  # 7-Dof Action Space
            # nn.Tanh(),
        )

    def forward(self, latent_action_tokens, visual_embed, proprio):
        proprio = self.proprio_proj(proprio)
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(
            torch.cat(
                [
                    self.attn_pool(
                        latent_action_tokens, init_embed=visual_embed
                    ),
                    proprio,
                ],
                dim=-1,
            )
        )

        return action


class Wrapped_Model(torch.nn.Module):
    def __init__(self, vla, freeze_vla=False, window_size=12):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        self.action_decoder = ActionDecoder(window_size=window_size)

        if freeze_vla:
            self.vla.requires_grad_(False)

    def forward(self, batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                labels=batch['labels'],
                output_hidden_states=True,  # Return intermediate tokens of all layers
            )
        loss, loss_one_step, latent_action_tokens = (
            self.action_decoder_forward(batch, vla_output)
        )

        return vla_output, loss, loss_one_step, latent_action_tokens

    def action_decoder_forward(self, batch, slow_output):
        # Task and action latents
        visual_embed = slow_output.hidden_states[-1][
            :, : self.vla.vision_backbone.featurizer.patch_embed.num_patches
        ].to(torch.float)
        latent_tokens = slow_output.hidden_states[-1][
            :, self.vla.vision_backbone.featurizer.patch_embed.num_patches :
        ]
        action_gt = batch['labels'].to(latent_tokens.device)
        mask = action_gt > 32000

        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[
                mask[idx], :
            ]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(
            torch.float
        )

        pred_action = self.action_decoder(
            latent_action_tokens, visual_embed
        ).reshape(-1, self.window_size, 7)
        loss = torch.nn.functional.l1_loss(
            pred_action, batch['actions'], reduction='none'
        )
        loss_one_step = loss[:, 0].mean()
        loss = loss.mean()

        return loss, loss_one_step, latent_action_tokens


@dataclass
class FinetuneConfig:
    # Directory Paths
    data_root_dir: Path = Path(
        '/path/to/your/local/hdf5_data'
    )  # Path to Open-X dataset directory

    vla_path: str = (
        '/path/to/your/pretrained-univla-7b'  # Path to your local UniVLA path
    )
    lam_path: str = (
        'latent_action_model/logs/task_centric_lam_stage2/epoch=0-step=200000.ckpt'
    )
    dataset_name: str = (
        'real_world'  # Name of fine-tuning dataset (e.g., `droid_wipe`)
    )
    run_root_dir: Path = Path(
        'runs'
    )  # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path(
        'adapter-tmp'
    )  # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 4  # Fine-tuning batch size
    max_steps: int = 10000  # Max number of fine-tuning steps
    save_steps: int = 2500  # Interval for checkpoint saving
    learning_rate: float = 3.5e-4  # Fine-tuning learning rate
    grad_accumulation_steps: int = 2  # Gradient accumulation steps
    image_aug: bool = False  # Whether to train with image augmentations
    shuffle_buffer_size: int = (
        100_00  # Dataloader shuffle buffer size (can reduce if OOM)
    )
    save_latest_checkpoint_only: bool = (
        True  # Whether to save only one checkpoint per run and
    )
    #   continually overwrite the latest checkpoint
    #   (If False, saves all checkpoints)
    # LAM setting
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_num_latents: int = 32
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12
    window_size: int = 12

    freeze_vla: bool = False
    # LoRA Arguments
    use_lora: bool = True  # Whether to use LoRA fine-tuning
    lora_rank: int = 32  # Rank of LoRA weight matrix
    lora_dropout: float = 0.0  # Dropout applied to LoRA weights
    use_quantization: bool = (
        False  # Whether to 4-bit quantize VLA for LoRA fine-tuning
    )
    #   => CAUTION: Reduces memory but hurts performance

    # hdf5 data config
    camera_names: str = 'camera_high'

    # Tracking Parameters
    wandb_project: str = (
        'fientune-real-world'  # Name of W&B project to log to (use default!)
    )
    wandb_entity: str = 'opendrivelab'  # Name of entity to log under
    run_id_note: str | None = None  # Extra note for logging, Weights & Biases


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f'Fine-tuning UniVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`')

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert (
        torch.cuda.is_available()
    ), 'Fine-tuning assumes at least one GPU is available!'
    distributed_state = PartialState()

    if distributed_state.is_main_process:
        print('This is the main process (rank 0).')
    else:
        print(
            f'This is a worker process (rank {distributed_state.process_index}).'
        )

    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision='bf16', kwargs_handlers=[ddp_kwargs]
    )

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f'+b{cfg.batch_size * cfg.grad_accumulation_steps}'
        f'+lr-{cfg.learning_rate}'
    )
    if cfg.use_lora:
        exp_id += f'+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}'
    if cfg.use_quantization:
        exp_id += '+q-4bit'
    if cfg.run_id_note is not None:
        exp_id += f'--{cfg.run_id_note}'
    if cfg.image_aug:
        exp_id += '--image_aug'

    exp_id += f'=w-LowLevelDecoder-ws-{cfg.window_size}'

    # Start =>> Build Directories
    run_dir, adapter_dir = (
        cfg.run_root_dir / exp_id,
        cfg.adapter_tmp_dir / exp_id,
    )
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert (
            cfg.use_lora
        ), 'Quantized training only supported for LoRA fine-tuning!'
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register('openvla', OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(
        cfg.vla_path, trust_remote_code=True
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules='all-linear',
            init_lora_weights='gaussian',
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    wrapped_model = Wrapped_Model(
        vla=vla, freeze_vla=cfg.freeze_vla, window_size=cfg.window_size
    ).to(device_id)

    trainable_total_params = sum(
        p.numel() for p in wrapped_model.parameters() if p.requires_grad
    )
    print('Total Trainable Params: ', trainable_total_params)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [
        param for param in wrapped_model.parameters() if param.requires_grad
    ]
    optimizer = AdamW(
        trainable_params, lr=cfg.learning_rate, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(cfg.max_steps * 8 * 0.5), gamma=0.1
    )

    from latent_action_model.genie.modules.lam import (
        ControllableDINOLatentActionModel,
    )

    latent_action_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.0,
    )

    lam_ckpt = torch.load(cfg.lam_path)['state_dict']
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace('lam.', '')] = lam_ckpt[key]

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()

    dataset_paths = find_all_hdf5(cfg.data_root_dir)
    dataloader, stats = load_data_univla(
        dataset_paths,
        [cfg.camera_names],
        cfg.batch_size,
        action_tokenizer,
        processor,
        window_size=cfg.window_size,
        min_window_size=cfg.window_size,
        max_window_size=cfg.window_size,
        image_transform=processor.image_processor.apply_transform,
    )

    # save stats and key information
    stats_dir = os.path.join(cfg.data_root_dir, 'stats')
    if not os.path.isdir(stats_dir):
        os.makedirs(stats_dir)
    print(f'Saving stats into {stats_dir}...')
    stats_path = os.path.join(stats_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    wrapped_model, latent_action_model, optimizer, scheduler, dataloader = (
        accelerator.prepare(
            wrapped_model,
            latent_action_model,
            optimizer,
            scheduler,
            dataloader,
        )
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        # if accelerator.is_main_process:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f'ft+{exp_id}',
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        wrapped_model.train()
        optimizer.zero_grad()
        current_step = 0
        while current_step < cfg.max_steps:

            for batch_idx, batch in enumerate(dataloader):

                batch['initial_pixel_values'] = batch[
                    'initial_pixel_values'
                ].to(device_id)
                batch['target_pixel_values'] = batch['target_pixel_values'].to(
                    device_id
                )
                batch['pixel_values'] = (
                    batch['pixel_values'].to(torch.bfloat16).to(device_id)
                )
                batch['actions'] = batch['actions'].to(device_id)
                batch['proprio'] = batch['proprio'].to(device_id)

                ### [TODO] We construct latent action labels (also history latent actions) on-the-fly
                ### This is a work-round of potential CUDA conflict of calling models in dataloader
                if len(batch['initial_pixel_values_hist']) > 1:
                    batch['initial_pixel_values_hist'] = batch[
                        'initial_pixel_values_hist'
                    ].to(device_id)
                    batch['target_pixel_values_hist'] = batch[
                        'target_pixel_values_hist'
                    ].to(device_id)

                    with torch.no_grad():
                        video = torch.stack(
                            [
                                batch['initial_pixel_values'],
                                batch['target_pixel_values'],
                            ],
                            dim=1,
                        )
                        latent_action_idx_batch = (
                            latent_action_model.module.vq_encode(video)[
                                'indices'
                            ].squeeze()
                        )
                        video = torch.stack(
                            [
                                batch['initial_pixel_values_hist'],
                                batch['target_pixel_values_hist'],
                            ],
                            dim=1,
                        )
                        latent_action_idx_history = (
                            latent_action_model.module.vq_encode(video)[
                                'indices'
                            ].squeeze()
                        )

                    input_ids_list = []
                    labels_list = []
                    hist_idx = 0
                    # print(batch['with_hist'],latent_action_idx_history.shape)
                    for idx, latent_action_idx in enumerate(
                        latent_action_idx_batch
                    ):
                        action_vocab = [
                            f'<ACT_{i.item()}>' for i in latent_action_idx
                        ]  # [ACT_1, ACT_2, ... ACT_K]
                        action_tokens = ''
                        for i, action in enumerate(action_vocab):
                            action_tokens += action

                        if batch['with_hist'][idx]:
                            action_vocab = [
                                f'<ACT_{i.item()}>'
                                for i in latent_action_idx_history[hist_idx]
                            ]

                            hist_action_tokens = ''
                            for i, action in enumerate(action_vocab):
                                hist_action_tokens += action

                            input_prompt = (
                                f"What action should the robot take to {batch['instructions'][idx].lower()}? History action "
                                + hist_action_tokens
                            )
                            hist_idx += 1
                        else:
                            input_prompt = f"What action should the robot take to {batch['instructions'][idx].lower()}?"

                        # Add instruction to VLA prompt
                        prompt_builder = PurePromptBuilder('openvla')
                        conversation = [
                            {'from': 'human', 'value': input_prompt},
                            {'from': 'gpt', 'value': action_tokens},
                        ]
                        for turn in conversation:
                            prompt_builder.add_turn(
                                turn['from'], turn['value']
                            )

                        # Tokenize (w/ `base_tokenizer`)
                        input_ids = processor.tokenizer(
                            prompt_builder.get_prompt(),
                            add_special_tokens=True,
                        ).input_ids
                        labels = list(input_ids)

                        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
                        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
                        input_ids, labels = torch.tensor(
                            input_ids
                        ), torch.tensor(labels)

                        labels[: -(len(action_vocab) + 1)] = -100

                        input_ids_list.append(input_ids)
                        labels_list.append(labels)

                else:
                    with torch.no_grad():
                        video = torch.stack(
                            [
                                batch['initial_pixel_values'],
                                batch['target_pixel_values'],
                            ],
                            dim=1,
                        )
                        latent_action_idx_batch = (
                            latent_action_model.module.vq_encode(video)[
                                'indices'
                            ].squeeze()
                        )

                    input_ids_list = []
                    labels_list = []
                    for idx, latent_action_idx in enumerate(
                        latent_action_idx_batch
                    ):
                        action_vocab = [
                            f'<ACT_{i.item()}>' for i in latent_action_idx
                        ]  # [ACT_1, ACT_2, ... ACT_K]

                        action_tokens = ''
                        for i, action in enumerate(action_vocab):
                            action_tokens += action

                        # Add instruction to VLA prompt
                        prompt_builder = PurePromptBuilder('openvla')
                        conversation = [
                            {
                                'from': 'human',
                                'value': f"What action should the robot take to {batch['instructions'][idx].lower()}?",
                            },
                            {'from': 'gpt', 'value': action_tokens},
                        ]
                        for turn in conversation:
                            prompt_builder.add_turn(
                                turn['from'], turn['value']
                            )

                        # Tokenize (w/ `base_tokenizer`)
                        input_ids = processor.tokenizer(
                            prompt_builder.get_prompt(),
                            add_special_tokens=True,
                        ).input_ids
                        labels = list(input_ids)

                        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
                        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
                        input_ids, labels = torch.tensor(
                            input_ids
                        ), torch.tensor(labels)

                        labels[: -(len(action_vocab) + 1)] = -100

                        input_ids_list.append(input_ids)
                        labels_list.append(labels)

                input_ids = pad_sequence(
                    input_ids_list,
                    batch_first=True,
                    padding_value=processor.tokenizer.pad_token_id,
                )
                labels = pad_sequence(
                    labels_list, batch_first=True, padding_value=-100
                )

                # Truncate (if necessary)
                input_ids, labels = (
                    input_ids[:, : processor.tokenizer.model_max_length],
                    labels[:, : processor.tokenizer.model_max_length],
                )

                # Get `attention_mask` by checking for `pad_token_id`
                attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)

                batch['input_ids'] = input_ids
                batch['attention_mask'] = attention_mask
                batch['labels'] = labels

                output, act_loss, loss_one_step, latent_action_tokens = (
                    wrapped_model(batch)
                )

                loss = act_loss if cfg.freeze_vla else act_loss + output.loss
                # Normalize loss to account for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps

                torch.nn.utils.clip_grad_norm_(
                    wrapped_model.parameters(), max_norm=0.3
                )
                # Backward pass
                normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                action_logits = output.logits[
                    :,
                    wrapped_model.module.vla.vision_backbone.featurizer.patch_embed.num_patches : -1,
                ]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch['labels'][:, 1:].to(action_preds.device)
                mask = action_gt > 32000

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = (
                    correct_preds.sum().float() / mask.sum().float()
                )

                # Store recent train metrics
                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())

                # Compute gradient step index
                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

                # Compute smoothened train metrics
                #   =>> Equal to current step metrics when not using gradient accumulation
                #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(
                    recent_action_accuracies
                ) / len(recent_action_accuracies)

                # Push Metrics to W&B (every 10 gradient steps)
                if (
                    distributed_state.is_main_process
                ):  # and gradient_step_idx % 2 == 0:
                    # if accelerator.is_main_process and gradient_step_idx % 5 == 0:
                    # print("Step{}: Logging to wandb...".format(gradient_step_idx + current_step))
                    wandb.log(
                        {
                            'train_loss': smoothened_loss,
                            'action_accuracy': smoothened_action_accuracy,
                            'action_loss': act_loss.item(),
                            'action_loss_1step': loss_one_step.item(),
                            'lr': optimizer.state_dict()['param_groups'][0][
                                'lr'
                            ],
                            # "latent_align_loss": latent_align_loss.item(),
                        },
                        step=gradient_step_idx + current_step,
                    )

                # Optimizer Step
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    progress.update()

                # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                if (gradient_step_idx + current_step) > 0 and (
                    gradient_step_idx + current_step
                ) % cfg.save_steps == 0:
                    print(
                        f'This is a process (rank {distributed_state.process_index}).'
                    )
                    if distributed_state.is_main_process:
                        print(
                            f'Saving Model Checkpoint for Step {gradient_step_idx + current_step}'
                        )

                        # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                        save_dir = adapter_dir if cfg.use_lora else run_dir
                        save_dir = (
                            str(save_dir)
                            + f'/{gradient_step_idx + current_step}'
                        )

                        # Save Processor & Weights
                        if not cfg.freeze_vla:
                            processor.save_pretrained(
                                str(run_dir)
                                + f'/{gradient_step_idx + current_step}'
                            )
                            wrapped_model.module.vla.save_pretrained(save_dir)

                        dir_path = (
                            str(run_dir)
                            + f'/{gradient_step_idx + current_step}'
                        )
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        # Save low-level policy
                        torch.save(
                            wrapped_model.module.action_decoder.state_dict(),
                            str(run_dir)
                            + f'/{gradient_step_idx + current_step}'
                            + f'/action_decoder-{gradient_step_idx + current_step}.pt',
                        )

                    # Wait for processor and adapter weights to be saved by main process
                    dist.barrier()

                    # Merge LoRA weights into model backbone for faster inference
                    #   =>> Note that merging is slow and can be done post-hoc to speed up training
                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path,
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                        )
                        # merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = PeftModel.from_pretrained(
                            base_vla,
                            str(adapter_dir)
                            + f'/{gradient_step_idx + current_step}',
                        )
                        merged_vla = merged_vla.merge_and_unload()
                        if distributed_state.is_main_process:
                            # if accelerator.is_main_process:
                            if cfg.save_latest_checkpoint_only:
                                # Overwrite latest checkpoint
                                merged_vla.save_pretrained(
                                    str(run_dir)
                                    + f'/{gradient_step_idx + current_step}'
                                )
                                print(
                                    f'Saved Model Checkpoint for Step {gradient_step_idx + current_step} at: {run_dir}/{gradient_step_idx + current_step}'
                                )
                            else:
                                # Prepare to save checkpoint in new directory
                                checkpoint_dir = Path(
                                    str(run_dir)
                                    + f'/{gradient_step_idx + current_step}'
                                    + f'--{gradient_step_idx + current_step}_chkpt'
                                )
                                os.makedirs(checkpoint_dir, exist_ok=True)

                                # Save processor and model weights to new directory
                                processor.save_pretrained(checkpoint_dir)
                                merged_vla.save_pretrained(checkpoint_dir)

                                print(
                                    f'Saved Model Checkpoint for Step {gradient_step_idx + current_step} at: {checkpoint_dir}'
                                )

                    # Block on Main Process Checkpointing
                    dist.barrier()

            description = f'Epoch {current_step + 1} | action_loss: {act_loss.item():.4f} | acc: {smoothened_action_accuracy:.4f}'
            progress.set_description(description)

            current_step = gradient_step_idx + 1 + current_step
            # Stop training when max_steps is reached
            if current_step >= cfg.max_steps:
                print(
                    f'Max step {cfg.max_steps} reached! Stopping training...'
                )
                wandb.finish()
                break


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn', force=True)
    finetune()
