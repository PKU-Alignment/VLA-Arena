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
vla.py

Draccus Dataclass Definition for a VLAConfig object, with various registered subclasses for each VLA experiment and
model configuration thereof. A given VLA model (`policy`) configures the following attributes:
    - Data Mixture (e.g., Bridge, OXE_MAGIC_SOUP, etc.)
    - Base VLM from Prismatic Registry (e.g., `prism-dinosiglip+7b`)
    - VLA Model Architecture / Parameters (e.g., freeze vision encoder, last layer finetuning)
    - Training / Optimization Hyperparameters
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path

from draccus import ChoiceRegistry


@dataclass
class VLAConfig(ChoiceRegistry):
    # fmt: off
    vla_id: str                                     # Unique VLA Policy ID that fully specifies a configuration variant
    base_vlm: str | Path                      # Base VLM as ID/Path to Run Directory (e.g., `prism-dinosiglip+7b`)
    freeze_vision_backbone: bool                    # Freeze Vision Backbone Parameters (akin to pretraining)
    freeze_llm_backbone: bool                       # Freeze LLM Backbone parameters
    unfreeze_last_llm_layer: bool                   # Unfreeze final layer of LLM (only takes effect if LLM is frozen)

    # Data Mixture Parameters
    data_mix: str                                   # Open-X Embodiment Dataset =>> Unique Mixture ID (e.g., `bridge`)
    shuffle_buffer_size: int                        # Size of Shuffle Buffer (100K for Bridge, 1M for OXE)

    # Optimization Parameters
    epochs: int                                     # Epochs to Run (in case `max_steps` is not specified)
    max_steps: int | None                        # [Optional] Max Gradient Steps to Run (overrides `epochs`)

    expected_world_size: int                        # Expected # of GPUs =>> allows us to gate training on hardware
    global_batch_size: int                          # Global Batch Size (divided across processes / world size)
    per_device_batch_size: int                      # Per-Device Batch Size (per-process / individual GPU)
                                                    #   =>> # of accumulation steps is auto-computed

    learning_rate: float                            # Peak Learning Rate (`lr_scheduler_type` sets warmup/decay)
    weight_decay: float                             # Weight Decay for AdamW Optimizer
    max_grad_norm: float                            # Max Grad Norm (for global gradient clipping)
    lr_scheduler_type: str                          # LR Scheduler (usually: "constant" | "linear-warmup+cosine-decay")
    warmup_ratio: float                             # Fraction of Steps to Warmup (for warmup LR schedulers)

    train_strategy: str                             # Train Strategy (default "fsdp-full-shard")

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True      # Enable Gradient/Activation Checkpointing during Training

    # Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True    # Enable Traditional BF16 Mixed Precision
    reduce_in_full_precision: bool = True           # Accumulate/Reduce All-Gather Gradients in FP32 Full Precision

    # fmt: on


# === OpenVLA Training Configurations ===


# = [8 GPU] Fast Iteration =>> SigLIP 224px + Bridge =
@dataclass
class Exp_SigLIP_224px_Bridge(VLAConfig):
    vla_id: str = 'siglip-224px+mx-bridge'
    base_vlm: str | Path = 'siglip-224px+7b'

    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = True

    # Data Mixture Parameters
    data_mix: str = 'oxe_magic_soup_plus'
    shuffle_buffer_size: int = 20_000

    # Optimization Parameters
    epochs: int = 10
    max_steps: int | None = None

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = 'constant'
    warmup_ratio: float = 0.0

    train_strategy: str = 'fsdp-full-shard'


# = [8 GPU] Fast Iteration =>> DINO-SigLIP 224px + Bridge =
@dataclass
class Exp_DinoSigLIP_224px_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'prism-dinosiglip-224px+mx-bridge'
    base_vlm: str | Path = 'prism-dinosiglip-224px+7b'

    data_mix: str = 'bridge'


@dataclass
class Exp_DinoSigLIP_224px_Human(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'prism-dinosiglip-224px+mx-human'
    base_vlm: str | Path = 'prism-dinosiglip-224px+7b'

    data_mix: str = 'Ego4D'


# = [32 GPU Pre-training] DINO-SigLIP 224px + Magic Soup++ =
@dataclass
class Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus(Exp_SigLIP_224px_Bridge):
    vla_id: str = 'prism-dinosiglip-224px+mx-oxe-magic-soup-plus'
    base_vlm: str | Path = 'prism-dinosiglip-224px+7b'

    data_mix: str = 'omni_magic_soup_plus'  # OpenX (Manipulation + Navigation)
    # data_mix: str = "omni_magic_soup_plus_plus"   # OpenX + Humam

    expected_world_size: int = 32
    global_batch_size: int = 1024
    per_device_batch_size: int = 32


# === Define a VLA Registry Enum for Reference & Validation ===
@unique
class VLARegistry(Enum):
    # Sanity Check Configurations =>> BridgeV2
    SIGLIP_224PX_MX_BRIDGE = Exp_SigLIP_224px_Bridge

    # Pre-training on Bridge-v2 data only
    DINOSIGLIP_224PX_MX_BRIDGE = Exp_DinoSigLIP_224px_Bridge

    # Pre-training on Human data only
    DINOSIGLIP_224PX_MX_HUMAN = Exp_DinoSigLIP_224px_Human

    # Pre-training on full dataset
    DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS = (
        Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus
    )

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# Register VLAs in Choice Registry
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
