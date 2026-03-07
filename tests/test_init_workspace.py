from __future__ import annotations

from pathlib import Path

import pytest

from scripts import init_workspace


def test_initialize_workspace_creates_envs_and_configs(tmp_path: Path):
    init_workspace.initialize_workspace(
        output=tmp_path,
        models=['openvla', 'openpi'],
        copy_configs=True,
        force=False,
    )

    base_text = (tmp_path / 'envs/base/pyproject.toml').read_text(
        encoding='utf-8'
    )
    openvla_text = (tmp_path / 'envs/openvla/pyproject.toml').read_text(
        encoding='utf-8'
    )
    openpi_text = (tmp_path / 'envs/openpi/pyproject.toml').read_text(
        encoding='utf-8'
    )

    assert '"vla-arena",' in base_text
    assert 'vla-arena[openvla]' in openvla_text
    assert 'tool.uv.sources' not in openvla_text
    assert (
        'override-dependencies = ["ml-dtypes==0.5.4", "tensorstore==0.1.74"]'
        in openpi_text
    )
    assert (tmp_path / 'configs/train/openvla.yaml').exists()
    assert (tmp_path / 'configs/evaluation/openpi.yaml').exists()


def test_initialize_workspace_without_configs(tmp_path: Path):
    init_workspace.initialize_workspace(
        output=tmp_path,
        models=['openvla'],
        copy_configs=False,
        force=False,
    )
    assert (tmp_path / 'envs/openvla/pyproject.toml').exists()
    assert not (tmp_path / 'configs').exists()


def test_initialize_workspace_force_overwrites_existing_envs(tmp_path: Path):
    envs_dir = tmp_path / 'envs'
    envs_dir.mkdir(parents=True, exist_ok=True)
    (envs_dir / 'stale.txt').write_text('old', encoding='utf-8')

    with pytest.raises(FileExistsError, match='Target already exists'):
        init_workspace.initialize_workspace(
            output=tmp_path,
            models=['openvla'],
            copy_configs=False,
            force=False,
        )

    init_workspace.initialize_workspace(
        output=tmp_path,
        models=['openvla'],
        copy_configs=False,
        force=True,
    )
    assert not (envs_dir / 'stale.txt').exists()
    assert (tmp_path / 'envs/openvla/pyproject.toml').exists()
