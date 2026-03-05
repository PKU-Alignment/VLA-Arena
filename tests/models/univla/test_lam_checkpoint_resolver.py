"""Unit tests for UniVLA LAM checkpoint resolver."""

from unittest.mock import patch

import pytest

from vla_arena.models.univla.lam_checkpoint_resolver import (
    extract_lam_state_dict,
    resolve_lam_checkpoint,
    resolve_lam_checkpoint_path,
)


class _TensorLike:
    def __init__(self) -> None:
        self.shape = (1,)


def test_resolve_local_file_directly(tmp_path):
    ckpt_file = tmp_path / 'epoch=0-step=1.ckpt'
    ckpt_file.write_text('placeholder')

    resolved = resolve_lam_checkpoint(str(ckpt_file))

    assert resolved.source == 'local_file'
    assert resolved.resolved_path == str(ckpt_file)
    assert resolve_lam_checkpoint_path(str(ckpt_file)) == str(ckpt_file)


def test_resolve_local_dir_with_explicit_ckpt_file(tmp_path):
    ckpt_file = tmp_path / 'task_centric_lam_stage2' / 'epoch=0-step=200000.ckpt'
    ckpt_file.parent.mkdir(parents=True)
    ckpt_file.write_text('placeholder')

    resolved = resolve_lam_checkpoint(
        lam_path=str(tmp_path),
        lam_ckpt_file='task_centric_lam_stage2/epoch=0-step=200000.ckpt',
    )

    assert resolved.source == 'local_dir'
    assert resolved.resolved_path == str(ckpt_file)


def test_local_auto_select_prefers_higher_stage(tmp_path):
    stage1 = tmp_path / 'lam_stage1' / 'epoch=0-step=999999.ckpt'
    stage2 = tmp_path / 'lam_stage2' / 'epoch=0-step=1.ckpt'
    stage1.parent.mkdir(parents=True)
    stage2.parent.mkdir(parents=True)
    stage1.write_text('placeholder')
    stage2.write_text('placeholder')

    resolved = resolve_lam_checkpoint(str(tmp_path))

    assert resolved.resolved_path == str(stage2)


def test_local_auto_select_prefers_higher_step_with_same_stage(tmp_path):
    low_step = tmp_path / 'lam_stage2' / 'epoch=0-step=100.ckpt'
    high_step = tmp_path / 'lam_stage2' / 'epoch=0-step=200.ckpt'
    low_step.parent.mkdir(parents=True)
    low_step.write_text('placeholder')
    high_step.write_text('placeholder')

    resolved = resolve_lam_checkpoint(str(tmp_path))

    assert resolved.resolved_path == str(high_step)


def test_local_auto_select_prefers_ckpt_over_pt_with_same_score(tmp_path):
    pt_file = tmp_path / 'lam_stage2' / 'epoch=0-step=200.pt'
    ckpt_file = tmp_path / 'lam_stage2' / 'epoch=0-step=200.ckpt'
    pt_file.parent.mkdir(parents=True)
    pt_file.write_text('placeholder')
    ckpt_file.write_text('placeholder')

    resolved = resolve_lam_checkpoint(str(tmp_path))

    assert resolved.resolved_path == str(ckpt_file)


def test_hf_explicit_file_uses_hf_hub_download():
    with patch(
        'vla_arena.models.univla.lam_checkpoint_resolver.hf_hub_download',
        return_value='/tmp/downloaded.ckpt',
    ) as mock_download, patch(
        'vla_arena.models.univla.lam_checkpoint_resolver.HfApi'
    ) as mock_hf_api:
        resolved = resolve_lam_checkpoint(
            lam_path='org/repo',
            lam_ckpt_file='path/to/epoch=0-step=200000.ckpt',
        )

    mock_hf_api.assert_not_called()
    mock_download.assert_called_once_with(
        repo_id='org/repo',
        filename='path/to/epoch=0-step=200000.ckpt',
        repo_type='model',
    )
    assert resolved.source == 'hf_repo'
    assert resolved.resolved_path == '/tmp/downloaded.ckpt'


def test_hf_auto_selects_best_checkpoint_file():
    with patch(
        'vla_arena.models.univla.lam_checkpoint_resolver.HfApi'
    ) as mock_hf_api, patch(
        'vla_arena.models.univla.lam_checkpoint_resolver.hf_hub_download',
        return_value='/tmp/auto-selected.ckpt',
    ) as mock_download:
        mock_hf_api.return_value.list_repo_files.return_value = [
            'README.md',
            'task_centric_lam_stage1/epoch=0-step=400000.ckpt',
            'task_centric_lam_stage2/epoch=0-step=100000.pt',
            'task_centric_lam_stage2/epoch=0-step=100000.ckpt',
        ]

        resolved = resolve_lam_checkpoint('org/repo')

    mock_hf_api.return_value.list_repo_files.assert_called_once_with(
        repo_id='org/repo',
        repo_type='model',
    )
    mock_download.assert_called_once_with(
        repo_id='org/repo',
        filename='task_centric_lam_stage2/epoch=0-step=100000.ckpt',
        repo_type='model',
    )
    assert (
        resolved.selected_checkpoint
        == 'task_centric_lam_stage2/epoch=0-step=100000.ckpt'
    )


def test_env_override_takes_priority(monkeypatch, tmp_path):
    env_ckpt = tmp_path / 'env-stage2.ckpt'
    env_ckpt.write_text('placeholder')
    monkeypatch.setenv('UNIVLA_LAM_PATH', str(env_ckpt))

    resolved = resolve_lam_checkpoint('org/repo')

    assert resolved.source == 'local_file'
    assert resolved.resolved_path == str(env_ckpt)
    assert resolved.env_overridden is True


def test_hf_repo_without_checkpoint_candidates_raises():
    with patch(
        'vla_arena.models.univla.lam_checkpoint_resolver.HfApi'
    ) as mock_hf_api:
        mock_hf_api.return_value.list_repo_files.return_value = [
            'README.md',
            'config.json',
        ]
        with pytest.raises(FileNotFoundError, match='lam_ckpt_file'):
            resolve_lam_checkpoint('org/repo')


def test_hf_explicit_file_download_failure_message_contains_target():
    with patch(
        'vla_arena.models.univla.lam_checkpoint_resolver.hf_hub_download',
        side_effect=RuntimeError('download failed'),
    ):
        with pytest.raises(
            RuntimeError,
            match='org/repo/path/to/epoch=0-step=200000.ckpt',
        ):
            resolve_lam_checkpoint(
                lam_path='org/repo',
                lam_ckpt_file='path/to/epoch=0-step=200000.ckpt',
            )


def test_extract_lam_state_dict_from_nested_state_dict():
    state_dict = {'lam.weight': _TensorLike()}
    extracted = extract_lam_state_dict({'state_dict': state_dict})

    assert extracted == state_dict


def test_extract_lam_state_dict_from_plain_state_dict():
    plain_state_dict = {'lam.weight': _TensorLike()}
    extracted = extract_lam_state_dict(plain_state_dict)

    assert extracted == plain_state_dict


def test_extract_lam_state_dict_invalid_structure_raises():
    with pytest.raises(ValueError, match='Unsupported LAM checkpoint format'):
        extract_lam_state_dict({'epoch': 1, 'step': 2})
