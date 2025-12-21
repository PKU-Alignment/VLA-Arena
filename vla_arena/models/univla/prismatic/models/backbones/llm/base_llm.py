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
base_llm.py

Abstract class definition of a large (autoregressive) language model backbone (LLM), with full annotations of class
methods, utility functions, and initialization logic.

We also define the generic HFLLMBackbone class here, providing a default interface for loading any HF
AutoModelForCausalLM (e.g., LLamaForCausalLM). In general, we make the assumption that any given LLM backbone implements
the AutoModelForCausalLM API (though we may add Seq2Seq models in the future).

We make this assumption to keep the LLM handling in this codebase relatively lightweight, and to inherit all the nice HF
utilities around different types of decoding/generation strategies.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from vla_arena.models.univla.prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
)
from vla_arena.models.univla.prismatic.overwatch import initialize_overwatch


# Suppress HF Deprecation Warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for arbitrary HF LLM Backbones ===
class LLMBackbone(nn.Module, ABC):
    def __init__(self, llm_backbone_id: str) -> None:
        super().__init__()
        self.identifier = llm_backbone_id

        # Instance attributes for an LLM Backbone
        self.llm: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizerBase = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None: ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the LLM given targets (labels), returning the scalar Cross-Entropy Loss"""
        raise NotImplementedError

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor: ...

    @property
    @abstractmethod
    def prompt_builder_fn(self) -> type[PromptBuilder]: ...

    @property
    @abstractmethod
    def transformer_layer_cls(self) -> type[nn.Module]: ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]: ...

    @property
    def embed_dim(self) -> int:
        return self.llm.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id


# === Abstract Base Class for Arbitrary HF Causal LLMs ===
class HFCausalLLMBackbone(LLMBackbone, ABC):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_family: str,
        llm_cls: type[PreTrainedModel],
        hf_hub_path: str,
        llm_max_length: int = 2048,
        hf_token: str | None = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = False,
    ) -> None:
        super().__init__(llm_backbone_id)
        self.llm_family = llm_family
        self.llm_max_length = llm_max_length
        self.inference_mode = inference_mode

        # Initialize LLM (downloading from HF Hub if necessary) --> `llm_cls` is the actual {Model}ForCausalLM class!
        #   => Note: We're eschewing use of the AutoModel API so that we can be more explicit about LLM-specific details
        if not self.inference_mode:
            overwatch.info(
                f'Loading [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]',
                ctx_level=1,
            )
            self.llm = llm_cls.from_pretrained(
                hf_hub_path,
                token=hf_token,
                use_flash_attention_2=(
                    use_flash_attention_2 if not self.inference_mode else False
                ),
                # The following parameters are set to prevent `UserWarnings` from HF; we want greedy decoding!
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        # [Contract] `inference_mode` means we're loading from a pretrained checkpoint; no need to load base weights!
        else:
            overwatch.info(
                f'Building empty [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]',
                ctx_level=1,
            )
            llm_config = AutoConfig.from_pretrained(
                hf_hub_path, token=hf_token
            )
            self.llm = llm_cls._from_config(llm_config)

        # Lightweight Handling (with extended explanation) for setting some LLM Parameters
        #   => Set `decoder.use_cache = False` --> incompatible with gradient checkpointing (+ training in general)
        #
        #      Reference: https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
        self.llm.config.use_cache = False if not self.inference_mode else True

        #   => Turns out that when gradient checkpointing is on and the underlying LLM has no "trainable" parameters
        #      (requires_grad is False), backprop will fail; setting `enable_input_requires_grad()` registers a new
        #      forward hook that fixes this =>> also totally safe for the "full finetuning" setting!
        if not self.inference_mode:
            self.llm.enable_input_require_grads()

        # Load (Fast) Tokenizer
        overwatch.info(
            f'Loading [bold]{llm_family}[/] (Fast) Tokenizer via the AutoTokenizer API',
            ctx_level=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_hub_path,
            model_max_length=self.llm_max_length,
            token=hf_token,
            padding_side='right',
        )

        # Validation =>> Our VLM logic currently operates under the assumption that the tokenization of a new input
        #                starts with a <BOS> token unless `add_special_tokens = False`; for these models, we empirically
        #                find that adding image patches *after* the BOS leads to much better performance.
        #
        # As a result we explicitly validate that a tokenizer conforms to the expected behavior; if you're reading this
        # line, it's probably because you're adding a new LLM with a different tokenizer behavior. If so, feel free to
        # override the `SPECIAL_CASES` set below, but make sure to make the appropriate changes in the `datasets.py`
        # and VLM `forward()` logic!
        SPECIAL_CASES = {
            # Phi-2 Tokenizer doesn't add any BOS tokens by default, and sets BOS == EOS == "<|endoftext|>"
            #   =>> We'll prepend BOS to first input (to play nicely with image token insertion logic; verified that
            #       this works well with base LLM generation.
            #   =>> Like Llama-2 Tokenizers -- we'll add a special PAD token for training purposes.
            'phi-2-3b',
        }
        if self.identifier in SPECIAL_CASES:
            return

        # Note =>> this assert should hold for all Llama-derived tokenizers (`LlamaTokenizerFast` ==> includes Mistral!
        assert (
            self.tokenizer('Test 123', add_special_tokens=True).input_ids[0]
            == self.tokenizer.bos_token_id
        ) and (
            self.tokenizer('Test 123', add_special_tokens=False).input_ids[0]
            != self.tokenizer.bos_token_id
        ), (
            f'Default Tokenizer of type `{type(self.tokenizer)}` does not automatically prefix inputs with BOS token!\n'
            'Please read the comment in `base_llm.py` for more information!'
        )

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a `transformer_auto_wrap_policy` where we wrap each instance of `self.transformer_layer_cls`"""
        transformer_block_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={self.transformer_layer_cls},
        )

        return transformer_block_policy

    def enable_gradient_checkpointing(self) -> None:
        """Dispatch to underlying LLM instance's `gradient_checkpointing_enable`; defined for all `PretrainedModel`."""
        self.llm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    # [Contract] Should match the `forward` call of the underlying `llm` instance!
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithPast:
        output: CausalLMOutputWithPast = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return output
