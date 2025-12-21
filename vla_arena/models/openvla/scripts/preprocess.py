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
preprocess.py

Core script for automatically downloading raw VLM pretraining datasets. Supports downloading the following datasets:
    - LLaVA v1.5 Datasets (for both training stages) [`llava-laion-cc-sbu-558k`, `llava-v1.5-instruct`]
        - Stage 1 :: Projection Matrix Alignment between Vision Encoder & Pretrained LLM on CC-3M-595K (Custom)
        - Stage 2 :: Projection & LLM Finetuning on LLaVa v1.5 Instruct (including various vision-language train sets)

By default, runs download & extraction automatically.

Run with: `python scripts/preprocess.py --dataset_id <DATASET_ID>`
"""

from dataclasses import dataclass
from pathlib import Path

import draccus

from vla_arena.models.openvla.vla_arena.models.openvla.prismatic.overwatch import (
    initialize_overwatch,
)
from vla_arena.models.openvla.vla_arena.models.openvla.prismatic.preprocessing import (
    convert_to_jpg,
    download_extract,
)


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PreprocessConfig:
    # fmt: off
    dataset_id: str = 'llava-v1.5-instruct'                     # Unique identifier for dataset to process (see above)
    root_dir: Path = Path('data')                               # Path to root directory for storing datasets

    # fmt: on


@draccus.wrap()
def preprocess(cfg: PreprocessConfig) -> None:
    overwatch.info(
        f"Downloading & Extracting `{cfg.dataset_id}` to `{cfg.root_dir / 'download'}"
    )
    download_extract(cfg.dataset_id, root_dir=cfg.root_dir)

    # Special Handling for OCR VQA Images (for `llava-v1.5-instruct`) --> convert GIFs/PNGs to JPG
    if cfg.dataset_id == 'llava-v1.5-instruct':
        convert_to_jpg(
            cfg.root_dir / 'download' / cfg.dataset_id / 'ocr_vqa' / 'images'
        )


if __name__ == '__main__':
    preprocess()
