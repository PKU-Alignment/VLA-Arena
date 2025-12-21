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

Factory class for initializing Open-X Embodiment dataset kwargs and other parameters; provides and exports functions for
clear control flow.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

from vla_arena.models.openvla_oft.prismatic.overwatch import (
    initialize_overwatch,
)
from vla_arena.models.openvla_oft.prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from vla_arena.models.openvla_oft.prismatic.vla.datasets.rlds.oxe.configs import (
    OXE_DATASET_CONFIGS,
    ActionEncoding,
)
from vla_arena.models.openvla_oft.prismatic.vla.datasets.rlds.oxe.transforms import (
    OXE_STANDARDIZATION_TRANSFORMS,
)


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def make_oxe_dataset_kwargs(
    dataset_name: str,
    data_root_dir: Path,
    load_camera_views: tuple[str] = ('primary',),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
) -> dict[str, Any]:
    """Generates config (kwargs) for given dataset from Open-X Embodiment."""
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
    if dataset_kwargs['action_encoding'] not in [
        ActionEncoding.EEF_POS,
        ActionEncoding.EEF_R6,
        ActionEncoding.JOINT_POS_BIMANUAL,
    ]:
        raise ValueError(
            f'Cannot load `{dataset_name}`; only EEF_POS & EEF_R6 & JOINT_POS_BIMANUAL actions supported!'
        )

    # [Contract] For EEF_POS & EEF_R6 actions, only the last action dimension (gripper) is absolute!
    # Normalize all action dimensions *except* the gripper
    if dataset_kwargs['action_encoding'] is ActionEncoding.EEF_POS:
        dataset_kwargs['absolute_action_mask'] = [False] * 6 + [True]
        dataset_kwargs['action_normalization_mask'] = [True] * 6 + [False]
    elif dataset_kwargs['action_encoding'] is ActionEncoding.EEF_R6:
        dataset_kwargs['absolute_action_mask'] = [False] * 9 + [True]
        dataset_kwargs['action_normalization_mask'] = [True] * 9 + [False]
    elif (
        dataset_kwargs['action_encoding'] is ActionEncoding.JOINT_POS_BIMANUAL
    ):
        dataset_kwargs['absolute_action_mask'] = [True] * 14
        dataset_kwargs['action_normalization_mask'] = [True] * 14
    dataset_kwargs['action_proprio_normalization_type'] = (
        action_proprio_normalization_type
    )

    # Adjust Loaded Camera Views
    if (
        len(
            missing_keys := (
                set(load_camera_views) - set(dataset_kwargs['image_obs_keys'])
            )
        )
        > 0
    ):
        raise ValueError(
            f'Cannot load `{dataset_name}`; missing camera views `{missing_keys}`'
        )

    # Filter
    dataset_kwargs['image_obs_keys'] = {
        k: v
        for k, v in dataset_kwargs['image_obs_keys'].items()
        if k in load_camera_views
    }
    dataset_kwargs['depth_obs_keys'] = {
        k: v
        for k, v in dataset_kwargs['depth_obs_keys'].items()
        if k in load_camera_views
    }

    # Eliminate Unnecessary Keys
    dataset_kwargs.pop('state_encoding')
    dataset_kwargs.pop('action_encoding')
    if not load_depth:
        dataset_kwargs.pop('depth_obs_keys')
    if not load_proprio:
        dataset_kwargs.pop('state_obs_keys')

    # Load Language
    if load_language:
        dataset_kwargs['language_key'] = 'language_instruction'

    # Specify Standardization Transform
    dataset_kwargs['standardize_fn'] = OXE_STANDARDIZATION_TRANSFORMS[
        dataset_name
    ]

    # Add any aux arguments
    if 'aux_kwargs' in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop('aux_kwargs'))

    return {
        'name': dataset_name,
        'data_dir': str(data_root_dir),
        **dataset_kwargs,
    }


def get_oxe_dataset_kwargs_and_weights(
    data_root_dir: Path,
    mixture_spec: list[tuple[str, float]],
    load_camera_views: tuple[str] = ('primary',),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
) -> tuple[dict[str, Any], list[float]]:
    """
    Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset. The returned kwargs
    (per-dataset configs) and weights can be passed directly to `make_interleaved_dataset`.

    :param data_root_dir: Base directory containing RLDS/TFDS-formatted datasets (from Open-X)
    :param mixture_spec: List of (dataset_name, sampling_weight) from `oxe.mixtures.OXE_NAMED_MIXTURES`
    :param load_camera_views: Camera views to load; see `oxe.dataset_configs.py` for available views.
    :param load_depth: Load depth information in addition to camera RGB.
    :param load_proprio: Load proprioceptive state.
    :param load_language: Load language instructions.
    :param action_proprio_normalization_type: Normalization scheme to use for proprioceptive actions.

    return: Tuple of (per_dataset_kwargs, sampling_weights)
    """
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight in mixture_spec:
        if d_name in included_datasets:
            overwatch.warning(
                f'Skipping Duplicate Dataset: `{(d_name, d_weight)}`'
            )
            continue

        included_datasets.add(d_name)
        filtered_mixture_spec.append((d_name, d_weight))

    # Assemble Dataset Config (kwargs) and Weights
    per_dataset_kwargs, sampling_weights = [], []
    for d_name, d_weight in filtered_mixture_spec:
        try:
            per_dataset_kwargs.append(
                make_oxe_dataset_kwargs(
                    d_name,
                    data_root_dir,
                    load_camera_views,
                    load_depth,
                    load_proprio,
                    load_language,
                    action_proprio_normalization_type,
                )
            )
            sampling_weights.append(d_weight)

        except ValueError as e:
            overwatch.warning(f'Skipping `{d_name}` due to Error: {e}')

    return per_dataset_kwargs, sampling_weights
