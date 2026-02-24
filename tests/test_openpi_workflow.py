from __future__ import annotations

import pickle
import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vla_arena.models.openpi import workflow_utils


class _DummyDataFactory:
    def __init__(self, data_config):
        self._data_config = data_config

    def create(self, *_args, **_kwargs):
        return self._data_config


class _DummyTrainConfig:
    def __init__(self, checkpoint_dir: pathlib.Path):
        self._checkpoint_dir = checkpoint_dir
        self.assets_dirs = checkpoint_dir.parent / 'assets'
        self.model = object()
        self.data = _DummyDataFactory(
            SimpleNamespace(repo_id='datasets/vla-arena', norm_stats=None)
        )

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return self._checkpoint_dir


def _make_step_dir(base: pathlib.Path, step: int) -> pathlib.Path:
    step_dir = base / str(step)
    (step_dir / 'params').mkdir(parents=True, exist_ok=True)
    return step_dir


def test_resolve_checkpoint_dir_explicit_step_dir(tmp_path: pathlib.Path):
    step_dir = _make_step_dir(tmp_path, 1000)
    resolved = workflow_utils.resolve_checkpoint_dir(
        step_dir, train_cfg=None, policy_checkpoint_step='latest'
    )
    assert pathlib.Path(resolved) == step_dir.resolve()


def test_resolve_checkpoint_dir_uses_explicit_path_over_train_cfg(
    tmp_path: pathlib.Path,
):
    explicit_exp_dir = tmp_path / 'explicit'
    _make_step_dir(explicit_exp_dir, 20)

    inferred_exp_dir = tmp_path / 'inferred'
    _make_step_dir(inferred_exp_dir, 99)
    train_cfg = _DummyTrainConfig(inferred_exp_dir)

    resolved = workflow_utils.resolve_checkpoint_dir(
        explicit_exp_dir, train_cfg=train_cfg, policy_checkpoint_step='latest'
    )
    assert pathlib.Path(resolved) == (explicit_exp_dir / '20').resolve()


def test_resolve_checkpoint_dir_from_train_cfg_uses_latest(
    tmp_path: pathlib.Path,
):
    exp_dir = tmp_path / 'exp'
    _make_step_dir(exp_dir, 5)
    _make_step_dir(exp_dir, 17)
    _make_step_dir(exp_dir, 101)
    train_cfg = _DummyTrainConfig(exp_dir)

    resolved = workflow_utils.resolve_checkpoint_dir(
        None, train_cfg=train_cfg, policy_checkpoint_step='latest'
    )
    assert pathlib.Path(resolved) == (exp_dir / '101').resolve()


def test_resolve_checkpoint_dir_raises_when_no_steps(tmp_path: pathlib.Path):
    exp_dir = tmp_path / 'empty_exp'
    exp_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match='No checkpoint step directories'):
        workflow_utils.resolve_checkpoint_dir(
            exp_dir, train_cfg=None, policy_checkpoint_step='latest'
        )


def test_ensure_norm_stats_skips_when_already_exists(monkeypatch):
    cfg = _DummyTrainConfig(pathlib.Path('/tmp/checkpoints/openpi/run'))
    cfg.data = _DummyDataFactory(
        SimpleNamespace(repo_id='datasets/vla-arena', norm_stats={'state': 1})
    )
    compute_mock = Mock()
    monkeypatch.setattr(
        workflow_utils, 'compute_and_save_norm_stats', compute_mock
    )

    workflow_utils.ensure_norm_stats(cfg)
    compute_mock.assert_not_called()


def test_ensure_norm_stats_computes_when_missing(monkeypatch):
    cfg = _DummyTrainConfig(pathlib.Path('/tmp/checkpoints/openpi/run'))
    missing_data_config = SimpleNamespace(
        repo_id='datasets/vla-arena',
        norm_stats=None,
    )
    loaded_data_config = SimpleNamespace(
        repo_id='datasets/vla-arena',
        norm_stats={'state': 1},
    )
    cfg.data = _DummyDataFactory(missing_data_config)

    def _compute_and_simulate_reload(_cfg, max_frames=None):
        del max_frames
        cfg.data = _DummyDataFactory(loaded_data_config)
        return pathlib.Path('/tmp/assets/datasets/vla-arena')

    compute_mock = Mock(side_effect=_compute_and_simulate_reload)
    monkeypatch.setattr(
        workflow_utils, 'compute_and_save_norm_stats', compute_mock
    )

    workflow_utils.ensure_norm_stats(cfg)
    compute_mock.assert_called_once()


def test_trainer_main_invokes_norm_stats_then_train_loop(monkeypatch):
    trainer = pytest.importorskip('vla_arena.models.openpi.trainer')
    cfg = trainer._config.get_config('debug')

    ensure_mock = Mock()
    train_loop_mock = Mock()
    monkeypatch.setattr(trainer, 'ensure_norm_stats', ensure_mock)
    monkeypatch.setattr(trainer, 'train_loop', train_loop_mock)

    trainer.main(config=cfg)
    ensure_mock.assert_called_once_with(cfg)
    train_loop_mock.assert_called_once_with(cfg)


def test_remove_strings_transform_is_picklable():
    transform = workflow_utils._RemoveStringsTransform()

    restored = pickle.loads(pickle.dumps(transform))
    assert isinstance(restored, workflow_utils._RemoveStringsTransform)


def test_remove_strings_transform_filters_string_fields():
    transform = workflow_utils._RemoveStringsTransform()
    item = {
        'state': np.asarray([1.0, 2.0]),
        'actions': np.asarray([[0.1, 0.2]]),
        'prompt': np.asarray('pick up the cup'),
        'task_name': 'stack blocks',
    }

    filtered = transform(item)
    assert 'state' in filtered
    assert 'actions' in filtered
    assert 'prompt' not in filtered
    assert 'task_name' not in filtered


def test_normalize_legacy_train_yaml_maps_checkpoint_path():
    yaml_data = {
        'name': 'pi0_vla_arena_low_mem_finetune',
        'weight_loader': {'checkpoint_path': '/tmp/params'},
    }

    normalized = workflow_utils._normalize_legacy_train_yaml(yaml_data)
    assert normalized['weight_loader']['params_path'] == '/tmp/params'
    assert 'checkpoint_path' not in normalized['weight_loader']


def test_normalize_legacy_train_yaml_prefers_params_path():
    yaml_data = {
        'name': 'pi0_vla_arena_low_mem_finetune',
        'weight_loader': {
            'checkpoint_path': '/tmp/legacy',
            'params_path': '/tmp/current',
        },
    }

    normalized = workflow_utils._normalize_legacy_train_yaml(yaml_data)
    assert normalized['weight_loader']['params_path'] == '/tmp/current'
    assert 'checkpoint_path' not in normalized['weight_loader']


def test_local_policy_client_reset_forwards_to_policy():
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')

    policy = Mock()
    client = evaluator._LocalPolicyClient(policy)
    client.reset()

    policy.reset.assert_called_once()


def test_safe_reset_policy_client_calls_reset():
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')

    client = Mock()
    evaluator._safe_reset_policy_client(client)

    client.reset.assert_called_once()


def test_run_task_cycles_initial_states(monkeypatch):
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')

    states = ['s0', 's1', 's2']
    captured_states = []

    class _DummyTaskSuite:
        def get_task_by_level_id(self, _task_level, _task_id):
            return SimpleNamespace(language='pick and place')

    cfg = SimpleNamespace(
        num_trials_per_task=5,
        add_noise=False,
        camera_offset=False,
        adjust_light=False,
        randomize_color=False,
        save_video_mode='none',
        seed=7,
        task_suite_name='safety_static_obstacles',
    )

    monkeypatch.setattr(
        evaluator,
        'load_initial_states',
        lambda *_args, **_kwargs: (states, None),
    )
    monkeypatch.setattr(
        evaluator,
        'get_vla_arena_env',
        lambda *_args, **_kwargs: (object(), 'pick and place'),
    )
    monkeypatch.setattr(evaluator.tqdm, 'tqdm', lambda it: it)

    def _fake_run_episode(
        _cfg,
        _env,
        _task_description,
        _replacements_dict,
        initial_state=None,
        log_file=None,
        client=None,
    ):
        del log_file, client
        captured_states.append(initial_state)
        return False, [], 0

    monkeypatch.setattr(evaluator, 'run_episode', _fake_run_episode)

    evaluator.run_task(
        cfg,
        _DummyTaskSuite(),
        task_id=0,
        task_level=0,
        replacements_dict={},
        total_episodes=0,
        total_successes=0,
        log_file=None,
        client=Mock(),
    )

    assert captured_states == ['s0', 's1', 's2', 's0', 's1']
