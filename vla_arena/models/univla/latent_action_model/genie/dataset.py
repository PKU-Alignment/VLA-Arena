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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torchvision.transforms as transforms
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from vla_arena.models.univla.vla_arena.models.univla.prismatic.util import (
    set_global_seed,
)
from vla_arena.models.univla.vla_arena.models.univla.prismatic.util.data_utils import (
    CollatorForLatentAction,
)
from vla_arena.models.univla.vla_arena.models.univla.prismatic.vla.datasets import (
    EpisodicRLDSDataset,
    RLDSBatchTransformVideo,
    RLDSDataset,
)


def exists(var) -> bool:
    return var is not None


def default(var, val) -> Any:
    return var if exists(var) else val


def default_worker_init_fn(worker_id: int) -> None:
    torch.manual_seed(torch.initial_seed() + worker_id)
    worker_info = get_worker_info()

    if exists(worker_info):
        dataset = worker_info.dataset
        glob_start = dataset._start
        glob_end = dataset._end

        per_worker = int((glob_end - glob_start) / worker_info.num_workers)
        worker_id = worker_info.id

        dataset._start = glob_start + worker_id * per_worker
        dataset._end = min(dataset._start + per_worker, glob_end)


class LightningDataset(LightningDataModule):
    """
    Abstract LightningDataModule that represents a dataset we can train a Lightning module on.
    """

    def __init__(
        self,
        *args,
        batch_size: int = 8,
        num_workers: int = 64,
        train_shuffle: bool = True,
        val_shuffle: bool = False,
        val_batch_size: int = None,
        worker_init_fn: Callable = None,
        collate_fn: Callable = None,
        train_sampler: Callable = None,
        test_sampler: Callable = None,
        val_sampler: Callable = None,
    ) -> None:
        super().__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers = 0  # For RLDS parallelism
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        # shuffle unspecified for iteratable datasets
        # self.train_shuffle = train_shuffle
        # self.val_shuffle = val_shuffle

        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(
                self.worker_init_fn, default_worker_init_fn
            )
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            # shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.val_dataset, IterableDataset):
            worker_init_fn = default(
                self.worker_init_fn, default_worker_init_fn
            )
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.val_batch_size,
            # shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.test_dataset, IterableDataset):
            worker_init_fn = default(
                self.worker_init_fn, default_worker_init_fn
            )
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.val_batch_size,
            # shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )


import random

from PIL import Image


@dataclass
class random_crop_resize:
    def __init__(self, target_size=224):
        self.target_size = target_size
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        width, height = image.size

        if width < height:
            crop_size = width
        else:
            crop_size = height

        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)

        image_cropped = image.crop(
            (left, top, left + crop_size, top + crop_size)
        )
        image_resized = image_cropped.resize(
            (self.target_size, self.target_size), Image.BILINEAR
        )
        image_resized = self.to_tensor(image_resized)

        return image_resized


class LightningOpenX(LightningDataset):
    """
    This dataset samples video recorded using a random agent
    playing the gym environments defined in the Procgen Benchmark,
    see Cobbe et al. ICML (2020).
    """

    def __init__(
        self,
        data_root: str,
        data_mix: str,
        batch_size: int = 16,
        resolution: int = 256,
        num_frames: int = 16,
        episodic: bool = False,
        shuffle_buffer_size: int = 100_000,
        image_aug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.data_root_dir = data_root
        self.data_mix = data_mix

        self.batch_size = batch_size
        self.resolution = (resolution, resolution)
        self.num_frames = num_frames

        self.episodic = episodic
        self.shuffle_buffer_size = shuffle_buffer_size
        self.image_aug = image_aug

        self.num_workers = 0  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        self.worker_init_fn = set_global_seed(42, get_worker_init_fn=True)

        self.batch_transform = RLDSBatchTransformVideo(
            image_transform=transforms.ToTensor()
        )
        self.collate_fn = CollatorForLatentAction()

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        cls = RLDSDataset if not self.episodic else EpisodicRLDSDataset
        if stage == 'fit':
            self.train_dataset = cls(
                self.data_root_dir,
                self.data_mix,
                self.batch_transform,
                resize_resolution=self.resolution,
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=True,
                image_aug=self.image_aug,
                training_phase='lam',
            )
            self.val_dataset = cls(
                self.data_root_dir,
                self.data_mix,
                self.batch_transform,
                resize_resolution=self.resolution,
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=False,
                image_aug=False,
                training_phase='lam',
            )
        elif stage == 'test':
            self.test_dataset = cls(
                self.data_root_dir,
                self.data_mix,
                self.batch_transform,
                resize_resolution=self.resolution,
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=True,
                image_aug=False,
                training_phase='lam',
            )
        else:
            raise ValueError(f'Invalid stage: {stage}')
