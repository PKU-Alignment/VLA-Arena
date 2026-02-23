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

from __future__ import annotations

import logging
import os
import pathlib
import sys
from typing import Any

# Add openpi src directory to Python path if needed.
_openpi_src = pathlib.Path(__file__).parent / 'src'
if str(_openpi_src) not in sys.path:
    sys.path.insert(0, str(_openpi_src))


class _RemoveStringsTransform:
    """Remove string-valued fields before computing normalization stats."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        return {
            k: v
            for k, v in data.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def _dict_to_tyro_args(prefix: str, data: dict[str, Any]) -> list[str]:
    """Recursively convert nested dict to tyro command-line args."""
    args = []
    for key, value in data.items():
        if key == 'name':
            continue
        full_key = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            args.extend(_dict_to_tyro_args(full_key, value))
        elif isinstance(value, (list, tuple)):
            args.append(f"--{full_key}={','.join(str(v) for v in value)}")
        elif isinstance(value, bool):
            # Keep behavior aligned with trainer: only emit True flags.
            if value:
                args.append(f'--{full_key}')
        elif value is None:
            continue
        else:
            args.append(f'--{full_key}={value}')
    return args


def _normalize_legacy_train_yaml(yaml_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy OpenPI YAML keys for backward compatibility."""
    normalized = dict(yaml_data)
    weight_loader = normalized.get('weight_loader')
    if not isinstance(weight_loader, dict):
        return normalized

    legacy_key = 'checkpoint_path'
    target_key = 'params_path'
    if legacy_key in weight_loader and target_key not in weight_loader:
        patched_weight_loader = dict(weight_loader)
        patched_weight_loader[target_key] = patched_weight_loader.pop(legacy_key)
        normalized['weight_loader'] = patched_weight_loader
        logging.warning(
            'Detected legacy key weight_loader.%s in train YAML. '
            'Auto-mapped to weight_loader.%s.',
            legacy_key,
            target_key,
        )
    elif legacy_key in weight_loader and target_key in weight_loader:
        patched_weight_loader = dict(weight_loader)
        patched_weight_loader.pop(legacy_key)
        normalized['weight_loader'] = patched_weight_loader
        logging.warning(
            'Both weight_loader.%s and weight_loader.%s are set in train YAML. '
            'Ignoring legacy key weight_loader.%s.',
            legacy_key,
            target_key,
            legacy_key,
        )
    return normalized


def load_train_config_from_yaml(
    config_path: str | pathlib.Path,
    override_kwargs: dict[str, Any] | None = None,
):
    """Load an OpenPI TrainConfig from a YAML file with overrides."""
    import openpi.training.config as _config
    import yaml

    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found at: {config_path}')

    with open(config_path) as f:
        yaml_data = yaml.safe_load(f)

    if isinstance(yaml_data, dict):
        yaml_data = _normalize_legacy_train_yaml(yaml_data)

    if override_kwargs:
        if not isinstance(yaml_data, dict):
            raise ValueError(
                f'Config file must contain a YAML dictionary, got {type(yaml_data)}'
            )
        yaml_data.update(override_kwargs)

    if not isinstance(yaml_data, dict) or 'name' not in yaml_data:
        raise ValueError(
            'OpenPI train config YAML must be a dictionary and contain "name".'
        )

    config_name = yaml_data['name']
    args_list = [config_name]
    args_list.extend(_dict_to_tyro_args('', yaml_data))

    original_argv = sys.argv.copy()
    try:
        sys.argv = ['openpi_train'] + args_list
        cfg = _config.cli()
    finally:
        sys.argv = original_argv
    return cfg


def _is_gcs_path(path: str) -> bool:
    return path.startswith('gs://')


def _list_checkpoint_steps(experiment_dir: pathlib.Path) -> list[int]:
    steps: list[int] = []
    for child in experiment_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        if (child / 'params').exists():
            steps.append(int(child.name))
    return sorted(steps)


def resolve_checkpoint_dir(
    policy_checkpoint_dir: str | pathlib.Path | None,
    train_cfg: Any | None,
    policy_checkpoint_step: str | int = 'latest',
) -> str | pathlib.Path:
    """Resolve checkpoint directory, supporting explicit path and latest-step fallback."""
    if policy_checkpoint_dir is not None:
        base = str(policy_checkpoint_dir)
    else:
        if train_cfg is None:
            raise ValueError(
                'Unable to resolve checkpoint path: both policy_checkpoint_dir and train config are missing.'
            )
        try:
            base = str(train_cfg.checkpoint_dir)
        except Exception as exc:
            raise ValueError(
                'Unable to infer checkpoint directory from train config. '
                'Please set policy_checkpoint_dir explicitly.'
            ) from exc

    if _is_gcs_path(base):
        if policy_checkpoint_step not in ('latest', None):
            return os.path.join(base.rstrip('/'), str(policy_checkpoint_step))
        return base

    base_path = pathlib.Path(base).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f'Checkpoint path does not exist: {base_path}')

    # Already a concrete checkpoint step dir.
    if (base_path / 'params').exists():
        return base_path

    # Otherwise treat this as experiment root and resolve step directory.
    if isinstance(policy_checkpoint_step, str):
        step_text = policy_checkpoint_step.strip().lower()
    else:
        step_text = str(policy_checkpoint_step)

    if step_text == 'latest':
        steps = _list_checkpoint_steps(base_path)
        if not steps:
            raise ValueError(
                f'No checkpoint step directories found under: {base_path}'
            )
        resolved = base_path / str(steps[-1])
        if not (resolved / 'params').exists():
            raise ValueError(
                f'Latest checkpoint directory does not contain params: {resolved}'
            )
        return resolved

    if not step_text.isdigit():
        raise ValueError(
            'policy_checkpoint_step must be an integer step or "latest".'
        )
    resolved = base_path / step_text
    if not (resolved / 'params').exists():
        raise FileNotFoundError(
            f'Checkpoint step directory not found or missing params: {resolved}'
        )
    return resolved


def _create_torch_norm_stats_dataloader(
    data_config,
    train_cfg,
    max_frames: int | None = None,
):
    import openpi.training.data_loader as _data_loader

    dataset = _data_loader.create_torch_dataset(
        data_config,
        train_cfg.model.action_horizon,
        train_cfg.model,
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _RemoveStringsTransform(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // train_cfg.batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // train_cfg.batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _create_rlds_norm_stats_dataloader(
    data_config,
    train_cfg,
    max_frames: int | None = None,
):
    import openpi.training.data_loader as _data_loader

    dataset = _data_loader.create_rlds_dataset(
        data_config,
        train_cfg.model.action_horizon,
        train_cfg.batch_size,
        shuffle=False,
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _RemoveStringsTransform(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // train_cfg.batch_size
    else:
        num_batches = len(dataset) // train_cfg.batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def compute_and_save_norm_stats(
    train_cfg,
    max_frames: int | None = None,
) -> pathlib.Path:
    """Compute and persist normalization stats for an OpenPI TrainConfig."""
    import numpy as np
    import openpi.shared.normalize as normalize
    import tqdm

    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_config.repo_id in (None, 'fake'):
        raise ValueError(
            f'Cannot compute normalization stats for repo_id={data_config.repo_id!r}.'
        )

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = _create_rlds_norm_stats_dataloader(
            data_config, train_cfg, max_frames
        )
    else:
        data_loader, num_batches = _create_torch_norm_stats_dataloader(
            data_config, train_cfg, max_frames
        )

    keys = ['state', 'actions']
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(
        data_loader, total=num_batches, desc='Computing norm stats'
    ):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = train_cfg.assets_dirs / data_config.repo_id
    normalize.save(output_path, norm_stats)
    logging.info('Saved OpenPI normalization stats to %s', output_path)
    return pathlib.Path(output_path)


def ensure_norm_stats(train_cfg, max_frames: int | None = None) -> pathlib.Path | None:
    """Ensure train config has norm stats. If missing, compute and save automatically."""
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    repo_id = data_config.repo_id
    if repo_id in (None, 'fake'):
        logging.info(
            'Skipping norm stats check for repo_id=%r (no normalization needed).',
            repo_id,
        )
        return None

    if data_config.norm_stats is not None:
        logging.info('Norm stats already available for repo_id=%s.', repo_id)
        return pathlib.Path(train_cfg.assets_dirs / repo_id)

    expected_path = pathlib.Path(train_cfg.assets_dirs) / repo_id
    logging.info(
        'Norm stats missing for repo_id=%s. Auto-computing at %s ...',
        repo_id,
        expected_path,
    )
    try:
        output_path = compute_and_save_norm_stats(train_cfg, max_frames=max_frames)
    except Exception as exc:
        raise RuntimeError(
            f'Failed to auto-compute normalization stats for repo_id={repo_id!r}. '
            f'Expected output directory: {expected_path}.'
        ) from exc

    refreshed_data_config = train_cfg.data.create(
        train_cfg.assets_dirs, train_cfg.model
    )
    if refreshed_data_config.norm_stats is None:
        raise RuntimeError(
            'Normalization stats were computed but could not be loaded from '
            f'{output_path}. Please verify `norm_stats.json` exists and is readable.'
        )
    return pathlib.Path(output_path)
