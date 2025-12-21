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

Factory class defining functions for instantiating various Training Strategies, supporting different VLMs, backbones,
and strategy configurations.
"""

from collections.abc import Callable

import torch

from vla_arena.models.univla.prismatic.models.vlms import PrismaticVLM
from vla_arena.models.univla.prismatic.training.strategies import (
    FSDPStrategy,
    TrainingStrategy,
)


# Registry =>> Maps ID --> {cls(), kwargs} :: supports FSDP for now, but DDP handler is also implemented!
TRAIN_STRATEGIES = {
    'fsdp-shard-grad-op': {
        'cls': FSDPStrategy,
        'kwargs': {'sharding_strategy': 'shard-grad-op'},
    },
    'fsdp-full-shard': {
        'cls': FSDPStrategy,
        'kwargs': {'sharding_strategy': 'full-shard'},
    },
}


def get_train_strategy(
    train_strategy: str,
    vlm: PrismaticVLM,
    device_id: int,
    stage: str,
    epochs: int,
    max_steps: int | None,
    global_batch_size: int,
    per_device_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    enable_gradient_checkpointing: bool = True,
    enable_mixed_precision_training: bool = True,
    reduce_in_full_precision: bool = False,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    worker_init_fn: Callable[[int], None] | None = None,
) -> TrainingStrategy:
    if train_strategy in TRAIN_STRATEGIES:
        strategy_cfg = TRAIN_STRATEGIES[train_strategy]
        strategy = strategy_cfg['cls'](
            vlm=vlm,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
            **strategy_cfg['kwargs'],
        )
        return strategy
    else:
        raise ValueError(
            f'Train Strategy `{train_strategy}` is not supported!'
        )
