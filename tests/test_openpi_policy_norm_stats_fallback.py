from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


policy_config = pytest.importorskip(
    'vla_arena.models.openpi.src.openpi.policies.policy_config'
)


def _make_norm_stats():
    return {
        'state': policy_config.transforms.NormStats(
            mean=np.asarray([0.0]),
            std=np.asarray([1.0]),
        ),
        'actions': policy_config.transforms.NormStats(
            mean=np.asarray([0.0]),
            std=np.asarray([1.0]),
        ),
    }


def test_resolve_policy_norm_stats_prefers_explicit(monkeypatch, tmp_path):
    data_config = SimpleNamespace(
        repo_id='repo/a', asset_id='asset_a', norm_stats=None
    )
    explicit_norm_stats = _make_norm_stats()
    load_mock = Mock(side_effect=AssertionError('should not be called'))
    monkeypatch.setattr(policy_config._checkpoints, 'load_norm_stats', load_mock)

    resolved = policy_config._resolve_policy_norm_stats(
        data_config, tmp_path, explicit_norm_stats
    )

    assert resolved is explicit_norm_stats
    load_mock.assert_not_called()


def test_resolve_policy_norm_stats_prefers_train_config(monkeypatch, tmp_path):
    train_norm_stats = _make_norm_stats()
    data_config = SimpleNamespace(
        repo_id='repo/a', asset_id='asset_a', norm_stats=train_norm_stats
    )
    load_mock = Mock(side_effect=AssertionError('should not be called'))
    monkeypatch.setattr(policy_config._checkpoints, 'load_norm_stats', load_mock)

    resolved = policy_config._resolve_policy_norm_stats(
        data_config, tmp_path, explicit_norm_stats=None
    )

    assert resolved is train_norm_stats
    load_mock.assert_not_called()


def test_resolve_policy_norm_stats_falls_back_to_exact_asset_id(
    monkeypatch, tmp_path, caplog
):
    data_config = SimpleNamespace(
        repo_id='repo/a', asset_id='asset_a', norm_stats=None
    )
    fallback_norm_stats = _make_norm_stats()

    load_mock = Mock(return_value=fallback_norm_stats)
    monkeypatch.setattr(policy_config._checkpoints, 'load_norm_stats', load_mock)
    caplog.set_level(logging.WARNING)

    resolved = policy_config._resolve_policy_norm_stats(
        data_config, tmp_path, explicit_norm_stats=None
    )

    assert resolved is fallback_norm_stats
    load_mock.assert_called_once_with(tmp_path / 'assets', 'asset_a')
    assert any(
        'Falling back to checkpoint assets' in record.message
        for record in caplog.records
    )
    assert any(
        'Using fallback norm stats from checkpoint assets' in record.message
        for record in caplog.records
    )


def test_resolve_policy_norm_stats_uses_recursive_unique_match(
    monkeypatch, tmp_path, caplog
):
    data_config = SimpleNamespace(
        repo_id='repo/a', asset_id='missing_asset', norm_stats=None
    )
    fallback_norm_stats = _make_norm_stats()
    unique_candidate = tmp_path / 'assets' / 'scan_asset' / 'norm_stats.json'
    unique_candidate.parent.mkdir(parents=True, exist_ok=True)
    unique_candidate.write_text('{}', encoding='utf-8')

    def _load_norm_stats(_assets_dir, asset_id):
        if asset_id == 'missing_asset':
            raise FileNotFoundError('missing exact asset')
        if asset_id == 'scan_asset':
            return fallback_norm_stats
        raise AssertionError(f'unexpected asset id: {asset_id}')

    load_mock = Mock(side_effect=_load_norm_stats)
    monkeypatch.setattr(policy_config._checkpoints, 'load_norm_stats', load_mock)
    caplog.set_level(logging.WARNING)

    resolved = policy_config._resolve_policy_norm_stats(
        data_config, tmp_path, explicit_norm_stats=None
    )

    assert resolved is fallback_norm_stats
    assert load_mock.call_count == 2
    assert any(
        'No norm stats at checkpoint exact fallback path' in record.message
        for record in caplog.records
    )
    assert any(
        'unique checkpoint asset match' in record.message
        for record in caplog.records
    )


def test_resolve_policy_norm_stats_raises_when_no_candidates(
    monkeypatch, tmp_path
):
    data_config = SimpleNamespace(
        repo_id='repo/a', asset_id='missing_asset', norm_stats=None
    )
    (tmp_path / 'assets').mkdir(parents=True, exist_ok=True)
    load_mock = Mock(side_effect=FileNotFoundError('missing exact asset'))
    monkeypatch.setattr(policy_config._checkpoints, 'load_norm_stats', load_mock)

    with pytest.raises(FileNotFoundError) as exc_info:
        policy_config._resolve_policy_norm_stats(
            data_config, tmp_path, explicit_norm_stats=None
        )

    message = str(exc_info.value)
    assert 'train config returned no norm stats' in message
    assert 'checkpoint exact fallback path tried' in message
    assert 'found 0 candidate files' in message
    assert str(tmp_path / 'assets' / 'missing_asset') in message
    assert 'Please set data.assets.asset_id' in message


def test_resolve_policy_norm_stats_raises_when_multiple_candidates(
    monkeypatch, tmp_path
):
    data_config = SimpleNamespace(
        repo_id='repo/a', asset_id='missing_asset', norm_stats=None
    )
    candidate_a = tmp_path / 'assets' / 'a' / 'norm_stats.json'
    candidate_b = tmp_path / 'assets' / 'b' / 'norm_stats.json'
    candidate_a.parent.mkdir(parents=True, exist_ok=True)
    candidate_b.parent.mkdir(parents=True, exist_ok=True)
    candidate_a.write_text('{}', encoding='utf-8')
    candidate_b.write_text('{}', encoding='utf-8')

    load_mock = Mock(side_effect=FileNotFoundError('missing exact asset'))
    monkeypatch.setattr(policy_config._checkpoints, 'load_norm_stats', load_mock)

    with pytest.raises(RuntimeError) as exc_info:
        policy_config._resolve_policy_norm_stats(
            data_config, tmp_path, explicit_norm_stats=None
        )

    message = str(exc_info.value)
    assert 'Unable to uniquely resolve OpenPI normalization stats' in message
    assert 'found 2 candidate files' in message
    assert str(candidate_a) in message
    assert str(candidate_b) in message
    assert 'Please set data.assets.asset_id' in message
