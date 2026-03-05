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

import argparse
import shutil
from pathlib import Path

from vla_arena.config_paths import get_packaged_configs_root


DEFAULT_MODELS = (
    'openvla',
    'openvla_oft',
    'univla',
    'smolvla',
    'openpi',
)

SUPPORTED_MODELS = frozenset(DEFAULT_MODELS)

_PROJECT_NAME_MAP = {
    'base': 'vla-arena-env-base',
    'openvla': 'vla-arena-env-openvla',
    'openvla_oft': 'vla-arena-env-openvla-oft',
    'univla': 'vla-arena-env-univla',
    'smolvla': 'vla-arena-env-smolvla',
    'openpi': 'vla-arena-env-openpi',
}

_EXTRA_MAP = {
    'openvla': 'openvla',
    'openvla_oft': 'openvla-oft',
    'univla': 'univla',
    'smolvla': 'smolvla',
    'openpi': 'openpi',
}


def _parse_models_csv(models_csv: str) -> list[str]:
    items = [part.strip() for part in models_csv.split(',') if part.strip()]
    if not items:
        raise ValueError('No models were provided in --models.')

    invalid = [model for model in items if model not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError(
            f'Unsupported model(s): {", ".join(invalid)}. '
            f'Supported: {", ".join(DEFAULT_MODELS)}'
        )

    # Keep input order while removing duplicates.
    seen = set()
    deduped = []
    for model in items:
        if model in seen:
            continue
        seen.add(model)
        deduped.append(model)
    return deduped


def _render_env_pyproject(model: str) -> str:
    if model == 'base':
        dependencies = '"vla-arena",'
    else:
        extra_name = _EXTRA_MAP[model]
        dependencies = f'"vla-arena",\n    "vla-arena[{extra_name}]",'

    lines = [
        '[project]',
        f'name = "{_PROJECT_NAME_MAP[model]}"',
        'version = "0.0.0"',
        'requires-python = "==3.11.*"',
        'dependencies = [',
        f'    {dependencies}',
        ']',
    ]
    if model == 'openpi':
        lines.extend(
            [
                '',
                '[tool.uv]',
                'override-dependencies = ["ml-dtypes==0.5.4", "tensorstore==0.1.74"]',
            ]
        )
    return '\n'.join(lines) + '\n'


def _prepare_output_dir(path: Path, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise FileExistsError(
            f'Target already exists: {path}. Use --force to overwrite.'
        )
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _write_env_projects(output_dir: Path, models: list[str], force: bool) -> None:
    envs_dir = output_dir / 'envs'
    _prepare_output_dir(envs_dir, force=force)
    envs_dir.mkdir(parents=True, exist_ok=True)

    ordered_models = ['base', *models]
    for model in ordered_models:
        model_dir = envs_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / 'pyproject.toml').write_text(
            _render_env_pyproject(model), encoding='utf-8'
        )


def _copy_configs(output_dir: Path, force: bool) -> None:
    source = get_packaged_configs_root()
    target = output_dir / 'configs'
    _prepare_output_dir(target, force=force)
    shutil.copytree(source, target)


def initialize_workspace(
    output: Path,
    models: list[str],
    copy_configs: bool,
    force: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write_env_projects(output_dir=output, models=models, force=force)
    if copy_configs:
        _copy_configs(output_dir=output, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Initialize local uv workspace projects for PyPI installation.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Output directory. Default: current directory.',
    )
    parser.add_argument(
        '--models',
        type=str,
        default=','.join(DEFAULT_MODELS),
        help='Comma-separated models. Default: openvla,openvla_oft,univla,smolvla,openpi',
    )
    parser.add_argument(
        '--without-configs',
        action='store_true',
        help='Only create envs/, skip copying configs/.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing envs/ and configs/ targets.',
    )
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    models = _parse_models_csv(args.models)
    include_configs = not args.without_configs

    initialize_workspace(
        output=output_dir,
        models=models,
        copy_configs=include_configs,
        force=args.force,
    )

    copied = 'yes' if include_configs else 'no'
    print(f'Workspace initialized at: {output_dir}')
    print(f'Model envs: base, {", ".join(models)}')
    print(f'Copied configs: {copied}')


if __name__ == '__main__':
    main()
