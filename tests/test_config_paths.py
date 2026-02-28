from __future__ import annotations

from pathlib import Path

import pytest

from vla_arena import config_paths


def test_resolve_default_train_config_path_for_openvla():
    resolved = config_paths.resolve_config_path(
        mode='train', model='openvla', config_path=None
    )
    resolved_path = Path(resolved)
    assert resolved_path.is_absolute()
    assert resolved_path.exists()
    assert resolved_path.name == 'openvla.yaml'
    assert resolved_path.parent.name == 'train'


def test_resolve_packaged_reference_without_repo_relative_file():
    resolved = config_paths.resolve_config_path(
        mode='eval',
        model='openvla',
        config_path='vla_arena/configs/evaluation/openvla.yaml',
    )
    resolved_path = Path(resolved)
    assert resolved_path.is_absolute()
    assert resolved_path.exists()
    assert resolved_path.name == 'openvla.yaml'
    assert resolved_path.parent.name == 'evaluation'


def test_resolve_config_path_requires_explicit_config_for_unknown_model():
    with pytest.raises(ValueError, match='No default train config'):
        config_paths.resolve_config_path(
            mode='train', model='unknown_model', config_path=None
        )


def test_resolve_config_path_missing_file_raises():
    with pytest.raises(FileNotFoundError, match='Config file not found'):
        config_paths.resolve_config_path(
            mode='train', model='openvla', config_path='not_found.yaml'
        )
