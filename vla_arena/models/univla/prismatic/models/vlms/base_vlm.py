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
base_vlm.py

Abstract class definition of a Vision-Language Model (VLM), with full annotations of class methods, utility functions,
and initialization logic. This is mostly to future-proof the codebase; while all our experiments instantiate
from PrismaticVLM, theoretically, this base class should be general enough to cover almost all models (e.g., IDEFICS,
PALI, Fuyu) in the future.

We use Abstract base classes *sparingly* -- mostly as a way to encapsulate any redundant logic or nested inheritance
(e.g., dependence on nn.Module, HF PretrainedModel, etc.). For other abstract objects (e.g., Tokenizers/Transforms),
prefer Protocol definitions instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from vla_arena.models.univla.prismatic.models.backbones.llm import LLMBackbone
from vla_arena.models.univla.prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
)
from vla_arena.models.univla.prismatic.models.backbones.vision import (
    VisionBackbone,
)


# === Abstract Base Class for arbitrary Vision-Language Models ===
class VLM(nn.Module, GenerationMixin, ABC):
    def __init__(
        self,
        model_family: str,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
    ) -> None:
        super().__init__()
        self.model_family, self.model_id = model_family, model_id
        self.vision_backbone, self.llm_backbone = vision_backbone, llm_backbone
        self.enable_mixed_precision_training = enable_mixed_precision_training

        # Instance Attributes for a generic VLM
        self.all_module_keys, self.trainable_module_keys = None, None

        # === GenerationMixin Expected Attributes =>> *DO NOT MODIFY* ===
        self.generation_config = self.llm_backbone.llm.generation_config
        self.main_input_name = 'input_ids'

    @property
    def device(self) -> torch.device:
        """Borrowed from `transformers.modeling_utils.py` -- checks parameter device; assumes model on *ONE* device!"""
        return next(self.parameters()).device

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_family: str,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        **kwargs: str,
    ) -> VLM: ...

    @abstractmethod
    def get_prompt_builder(
        self, system_prompt: str | None = None
    ) -> PromptBuilder: ...

    @abstractmethod
    def freeze_backbones(self, stage: str) -> None: ...

    @abstractmethod
    def load_from_checkpoint(
        self,
        stage: str,
        run_dir: Path,
        pretrained_checkpoint: Path | None = None,
    ) -> None: ...

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        multimodal_indices: torch.LongTensor | None = None,
    ) -> CausalLMOutputWithPast: ...

    # === GenerationMixin Expected Properties & Methods (DO NOT MODIFY) ===
    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def config(self) -> PretrainedConfig:
        return self.llm_backbone.llm.config

    # => Beam Search Utility
    def _reorder_cache(self, past_key_values, beam_idx):
        return self.llm_backbone.llm._reorder_cache(past_key_values, beam_idx)
