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
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""


from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from vla_arena.models.univla.prismatic.conf import DatasetConfig
from vla_arena.models.univla.prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
)
from vla_arena.models.univla.prismatic.models.backbones.vision import (
    ImageTransform,
)
from vla_arena.models.univla.prismatic.preprocessing.datasets import (
    AlignDataset,
    FinetuneDataset,
)
from vla_arena.models.univla.prismatic.util.data_utils import (
    PaddedCollatorForLanguageModeling,
)


# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {
    'align': AlignDataset,
    'finetune': FinetuneDataset,
    'full-finetune': FinetuneDataset,
}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: type[PromptBuilder],
    default_image_resolution: tuple[int, int, int],
    padding_side: str = 'right',
) -> tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        default_image_resolution,
        padding_side=padding_side,
    )

    # Switch on `stage`
    if stage == 'align':
        annotation_json, image_dir = dataset_cfg.align_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
        )
        return dataset, collator

    elif stage == 'finetune':
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    elif stage == 'full-finetune':
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f'Stage `{stage}` is not supported!')
