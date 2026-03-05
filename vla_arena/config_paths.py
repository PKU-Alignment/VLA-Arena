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

import importlib.resources
from pathlib import Path
from typing import Mapping


_PACKAGE_CONFIG_PREFIXES = (
    'vla_arena/configs/',
    './vla_arena/configs/',
)

_DEFAULT_CONFIGS: Mapping[str, Mapping[str, str]] = {
    'train': {
        'openvla': 'train/openvla.yaml',
        'openvla_oft': 'train/openvla_oft.yaml',
        'univla': 'train/univla.yaml',
        'smolvla': 'train/smolvla.yaml',
        'openpi': 'train/openpi.yaml',
    },
    'eval': {
        'openvla': 'evaluation/openvla.yaml',
        'openvla_oft': 'evaluation/openvla_oft.yaml',
        'univla': 'evaluation/univla.yaml',
        'smolvla': 'evaluation/smolvla.yaml',
        'openpi': 'evaluation/openpi.yaml',
    },
}


def _as_text(path_value: str | Path) -> str:
    return str(path_value).replace('\\', '/').strip()


def get_packaged_configs_root() -> Path:
    """Return the installed package's configs root path."""
    resource = importlib.resources.files('vla_arena').joinpath('configs')
    return Path(str(resource)).resolve()


def resolve_packaged_config_reference(config_path: str | Path) -> Path | None:
    """Resolve `vla_arena/configs/...` references inside installed package data."""
    text = _as_text(config_path)
    for prefix in _PACKAGE_CONFIG_PREFIXES:
        if not text.startswith(prefix):
            continue
        relative_path = text[len(prefix) :]
        candidate = (get_packaged_configs_root() / relative_path).resolve()
        if candidate.exists():
            return candidate
        return None
    return None


def resolve_config_path(
    mode: str, model: str, config_path: str | Path | None
) -> str:
    """Resolve user config path or model default config to an absolute file path."""
    if config_path is None:
        default_relative = _DEFAULT_CONFIGS.get(mode, {}).get(model)
        if default_relative is None:
            raise ValueError(
                f"No default {mode} config is available for model '{model}'. "
                'Please provide --config explicitly.'
            )
        candidate = (get_packaged_configs_root() / default_relative).resolve()
        if not candidate.exists():
            raise FileNotFoundError(
                f'Default {mode} config not found: {candidate}'
            )
        return str(candidate)

    raw = Path(str(config_path)).expanduser()
    if raw.exists():
        return str(raw.resolve())

    packaged = resolve_packaged_config_reference(raw)
    if packaged is not None:
        return str(packaged)

    raise FileNotFoundError(
        f'Config file not found: {config_path}. '
        "Pass an existing path or use 'vla_arena/configs/...'."
    )
