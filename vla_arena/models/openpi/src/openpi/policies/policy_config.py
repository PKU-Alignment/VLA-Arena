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

import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp
import vla_arena.models.openpi.src.openpi.models.model as _model
import vla_arena.models.openpi.src.openpi.policies.policy as _policy
import vla_arena.models.openpi.src.openpi.shared.download as download
import vla_arena.models.openpi.src.openpi.transforms as transforms
from vla_arena.models.openpi.src.openpi.training import checkpoints as _checkpoints
from vla_arena.models.openpi.src.openpi.training import config as _config


def _scan_norm_stats_files_in_assets(assets_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find all norm_stats.json files under a checkpoint assets directory."""
    if not assets_dir.exists():
        return []
    return sorted(assets_dir.rglob('norm_stats.json'))


def _resolve_policy_norm_stats(
    data_config: _config.DataConfig,
    checkpoint_dir: pathlib.Path,
    explicit_norm_stats: dict[str, transforms.NormStats] | None,
) -> dict[str, transforms.NormStats]:
    """Resolve norm stats with fallback: explicit -> train config -> checkpoint assets."""
    if explicit_norm_stats is not None:
        logging.info('Using explicitly provided norm stats for policy.')
        return explicit_norm_stats

    if data_config.norm_stats is not None:
        logging.info(
            'Using norm stats from train config (repo_id=%r, asset_id=%r).',
            data_config.repo_id,
            data_config.asset_id,
        )
        return data_config.norm_stats

    assets_dir = pathlib.Path(checkpoint_dir) / 'assets'
    exact_norm_stats_dir = (
        assets_dir / data_config.asset_id
        if data_config.asset_id is not None
        else None
    )
    logging.warning(
        'Norm stats missing in train config (repo_id=%r, asset_id=%r). '
        'Falling back to checkpoint assets at %s.',
        data_config.repo_id,
        data_config.asset_id,
        assets_dir,
    )

    if data_config.asset_id is not None:
        try:
            norm_stats = _checkpoints.load_norm_stats(
                assets_dir, data_config.asset_id
            )
            logging.warning(
                'Using fallback norm stats from checkpoint assets: %s',
                exact_norm_stats_dir,
            )
            return norm_stats
        except FileNotFoundError:
            logging.warning(
                'No norm stats at checkpoint exact fallback path: %s',
                exact_norm_stats_dir,
            )

    norm_stats_files = _scan_norm_stats_files_in_assets(assets_dir)
    if len(norm_stats_files) == 1:
        candidate_file = norm_stats_files[0]
        candidate_asset_id = candidate_file.parent.relative_to(
            assets_dir
        ).as_posix()
        norm_stats = _checkpoints.load_norm_stats(assets_dir, candidate_asset_id)
        logging.warning(
            'Using fallback norm stats from unique checkpoint asset match: %s',
            candidate_file,
        )
        return norm_stats

    train_source_desc = (
        'train config returned no norm stats '
        f'(repo_id={data_config.repo_id!r}, asset_id={data_config.asset_id!r}).'
    )
    exact_path_desc = (
        f'checkpoint exact fallback path tried: {exact_norm_stats_dir}.'
        if exact_norm_stats_dir is not None
        else 'checkpoint exact fallback path skipped because asset_id is None.'
    )
    guidance = (
        'Please set data.assets.asset_id to the correct checkpoint asset '
        'directory, or pass norm_stats explicitly to create_trained_policy().'
    )
    if not norm_stats_files:
        raise FileNotFoundError(
            'Unable to load OpenPI normalization stats. '
            f'{train_source_desc} {exact_path_desc} '
            f'Recursive scan under {assets_dir} found 0 candidate files. '
            f'{guidance}'
        )

    candidate_paths = ', '.join(str(path) for path in norm_stats_files)
    raise RuntimeError(
        'Unable to uniquely resolve OpenPI normalization stats from checkpoint '
        'assets. '
        f'{train_source_desc} {exact_path_desc} '
        f'Recursive scan under {assets_dir} found {len(norm_stats_files)} '
        f'candidate files: [{candidate_paths}]. {guidance}'
    )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, 'model.safetensors')
    is_pytorch = os.path.exists(weight_path)

    logging.info('Loading model...')
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
    else:
        model = train_config.model.load(
            _model.restore_params(
                checkpoint_dir / 'params', dtype=jnp.bfloat16
            )
        )
    data_config = train_config.data.create(
        train_config.assets_dirs, train_config.model
    )
    norm_stats = _resolve_policy_norm_stats(
        data_config,
        checkpoint_dir=checkpoint_dir,
        explicit_norm_stats=norm_stats,
    )

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            pytorch_device = 'cpu'

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
