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

"""RoboArena baseline policy configs."""

from typing import TypeAlias

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.droid_policy as droid_policy
import openpi.transforms as _transforms


ModelType: TypeAlias = _model.ModelType


def get_roboarena_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import (
        AssetsConfig,
        DataConfig,
        SimpleDataConfig,
        TrainConfig,
    )

    return [
        #
        # RoboArena DROID baseline inference configs.
        #
        TrainConfig(
            # Trained from PaliGemma, using RT-2 / OpenVLA style binning tokenizer.
            name='paligemma_binning_droid',
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                max_token_len=400,
                fast_model_tokenizer=_tokenizer.BinningTokenizer,
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id='droid'),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[
                        droid_policy.DroidInputs(
                            action_dim=model.action_dim,
                            model_type=ModelType.PI0_FAST,
                        )
                    ],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # Trained from PaliGemma, using FAST tokenizer (using universal FAST+ tokenizer).
            name='paligemma_fast_droid',
            model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=15),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id='droid'),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[
                        droid_policy.DroidInputs(
                            action_dim=model.action_dim,
                            model_type=ModelType.PI0_FAST,
                        )
                    ],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # Trained from PaliGemma, using FAST tokenizer (tokenizer trained on DROID dataset).
            name='paligemma_fast_specialist_droid',
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                fast_model_tokenizer=_tokenizer.FASTTokenizer,
                fast_model_tokenizer_kwargs={
                    'fast_tokenizer_path': 'KarlP/fast_droid_specialist'
                },
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id='droid'),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[
                        droid_policy.DroidInputs(
                            action_dim=model.action_dim,
                            model_type=ModelType.PI0_FAST,
                        )
                    ],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # Trained from PaliGemma, using FSQ tokenizer.
            name='paligemma_vq_droid',
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                fast_model_tokenizer=_tokenizer.FSQTokenizer,
                fast_model_tokenizer_kwargs={
                    'fsq_tokenizer_path': 'gs://openpi-assets/tokenizers/droid_fsq_tokenizer'
                },
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id='droid'),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[
                        droid_policy.DroidInputs(
                            action_dim=model.action_dim,
                            model_type=ModelType.PI0_FAST,
                        )
                    ],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # pi0-style diffusion / flow VLA, trained on DROID from PaliGemma.
            name='paligemma_diffusion_droid',
            model=pi0_config.Pi0Config(action_horizon=10, action_dim=8),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id='droid'),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[
                        droid_policy.DroidInputs(action_dim=model.action_dim)
                    ],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
    ]
