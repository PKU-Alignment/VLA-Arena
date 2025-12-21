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
datasets.py

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""

import os
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path

from draccus import ChoiceRegistry


def get_default_dataset_root() -> Path:
    """Get the default dataset root directory from environment variable or use a generic default."""
    default_root = os.environ.get(
        'PRISMATIC_DATASET_ROOT',
        os.environ.get('DATASET_ROOT', './datasets/prismatic-vlms'),
    )
    return Path(default_root)


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: tuple[Path, Path]       # Path to annotation file and images directory for `align` stage
    finetune_stage_components: tuple[Path, Path]    # Path to annotation file and images directory for `finetune` stage

    dataset_root_dir: Path                          # Path to dataset root directory; others paths are relative to root
    # fmt: on


# [Reproduction] LLaVa-v15 (exact dataset used in all public LLaVa-v15 models)
@dataclass
class LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = 'llava-v15'

    align_stage_components: tuple[Path, Path] = (
        Path('download/llava-laion-cc-sbu-558k/chat.json'),
        Path('download/llava-laion-cc-sbu-558k/'),
    )
    finetune_stage_components: tuple[Path, Path] = (
        Path('download/llava-v1.5-instruct/llava_v1_5_mix665k.json'),
        Path('download/llava-v1.5-instruct/'),
    )
    dataset_root_dir: Path = get_default_dataset_root()


# [Multimodal-Only] LLava-v15 WITHOUT the Language-Only ShareGPT Data (No Co-Training)
@dataclass
class LLaVa_Multimodal_Only_Config(DatasetConfig):
    dataset_id: str = 'llava-multimodal'

    align_stage_components: tuple[Path, Path] = (
        Path('download/llava-laion-cc-sbu-558k/chat.json'),
        Path('download/llava-laion-cc-sbu-558k/'),
    )
    finetune_stage_components: tuple[Path, Path] = (
        Path('download/llava-v1.5-instruct/llava_v1_5_stripped625k.json'),
        Path('download/llava-v1.5-instruct/'),
    )
    dataset_root_dir: Path = get_default_dataset_root()


# LLaVa-v15 + LVIS-Instruct-4V
@dataclass
class LLaVa_LVIS4V_Config(DatasetConfig):
    dataset_id: str = 'llava-lvis4v'

    align_stage_components: tuple[Path, Path] = (
        Path('download/llava-laion-cc-sbu-558k/chat.json'),
        Path('download/llava-laion-cc-sbu-558k/'),
    )
    finetune_stage_components: tuple[Path, Path] = (
        Path('download/llava-v1.5-instruct/llava_v1_5_lvis4v_mix888k.json'),
        Path('download/llava-v1.5-instruct/'),
    )
    dataset_root_dir: Path = get_default_dataset_root()


# LLaVa-v15 + LRV-Instruct
@dataclass
class LLaVa_LRV_Config(DatasetConfig):
    dataset_id: str = 'llava-lrv'

    align_stage_components: tuple[Path, Path] = (
        Path('download/llava-laion-cc-sbu-558k/chat.json'),
        Path('download/llava-laion-cc-sbu-558k/'),
    )
    finetune_stage_components: tuple[Path, Path] = (
        Path('download/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json'),
        Path('download/llava-v1.5-instruct/'),
    )
    dataset_root_dir: Path = get_default_dataset_root()


# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = 'llava-lvis4v-lrv'

    align_stage_components: tuple[Path, Path] = (
        Path('download/llava-laion-cc-sbu-558k/chat.json'),
        Path('download/llava-laion-cc-sbu-558k/'),
    )
    finetune_stage_components: tuple[Path, Path] = (
        Path(
            'download/llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_mix1231k.json'
        ),
        Path('download/llava-v1.5-instruct/'),
    )
    dataset_root_dir: Path = get_default_dataset_root()


# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    # === LLaVa v1.5 ===
    LLAVA_V15 = LLaVa_V15_Config

    LLAVA_MULTIMODAL_ONLY = LLaVa_Multimodal_Only_Config

    LLAVA_LVIS4V = LLaVa_LVIS4V_Config
    LLAVA_LRV = LLaVa_LRV_Config

    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(
        dataset_variant.dataset_id, dataset_variant.value
    )
