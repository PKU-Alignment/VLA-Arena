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


def test_is_local_host_variants():
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')

    assert evaluator._is_local_host('0.0.0.0')
    assert evaluator._is_local_host('127.0.0.1')
    assert evaluator._is_local_host('localhost')
    assert evaluator._is_local_host('::1')
    assert evaluator._is_local_host('ws://localhost')
    assert not evaluator._is_local_host('192.168.1.100')
    assert not evaluator._is_local_host('example.com')


def test_create_policy_client_reuses_existing_websocket_server(monkeypatch):
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')

    cfg = evaluator.GenerateConfig(
        host='127.0.0.1',
        port=8000,
        auto_start_policy_server=True,
    )
    client_obj = Mock()
    start_mock = Mock()
    ws_ctor = Mock(return_value=client_obj)

    monkeypatch.setattr(
        evaluator,
        '_resolve_policy_target',
        lambda _cfg: (object(), '/tmp/checkpoints/openpi/1000', 'pi0_cfg'),
    )
    monkeypatch.setattr(
        evaluator,
        '_is_port_open',
        Mock(return_value=True),
    )
    monkeypatch.setattr(
        evaluator, '_start_policy_server_process', start_mock
    )
    monkeypatch.setattr(
        evaluator._websocket_client_policy, 'WebsocketClientPolicy', ws_ctor
    )

    client, source, config_name, managed_process = evaluator._create_policy_client(
        cfg
    )

    assert client is client_obj
    assert source == '127.0.0.1:8000'
    assert config_name == 'pi0_cfg'
    assert managed_process is None
    start_mock.assert_not_called()
    ws_ctor.assert_called_once_with('127.0.0.1', 8000)


def test_create_policy_client_autostarts_server_when_local_port_unavailable(
    monkeypatch,
):
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')

    cfg = evaluator.GenerateConfig(
        host='localhost',
        port=8000,
        auto_start_policy_server=True,
    )
    managed_process = Mock()
    managed_process.pid = 12345
    managed_process.poll.return_value = None
    client_obj = Mock()

    monkeypatch.setattr(
        evaluator,
        '_resolve_policy_target',
        lambda _cfg: (object(), '/tmp/checkpoints/openpi/1000', 'pi0_cfg'),
    )
    monkeypatch.setattr(
        evaluator,
        '_is_port_open',
        Mock(return_value=False),
    )
    monkeypatch.setattr(
        evaluator,
        '_build_serve_policy_command',
        lambda *_args, **_kwargs: ['python', 'serve_policy.py'],
    )
    start_mock = Mock(return_value=managed_process)
    wait_mock = Mock()
    ws_ctor = Mock(return_value=client_obj)
    monkeypatch.setattr(
        evaluator, '_start_policy_server_process', start_mock
    )
    monkeypatch.setattr(
        evaluator, '_wait_for_policy_server_ready', wait_mock
    )
    monkeypatch.setattr(
        evaluator._websocket_client_policy, 'WebsocketClientPolicy', ws_ctor
    )

    client, source, config_name, process = evaluator._create_policy_client(cfg)

    assert client is client_obj
    assert source == 'localhost:8000'
    assert config_name == 'pi0_cfg'
    assert process is managed_process
    start_mock.assert_called_once()
    wait_mock.assert_called_once()
    ws_ctor.assert_called_once_with('localhost', 8000)


def test_create_policy_client_remote_host_unavailable_raises(monkeypatch):
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')

    cfg = evaluator.GenerateConfig(
        host='10.0.0.8',
        port=8000,
        auto_start_policy_server=True,
    )
    start_mock = Mock()
    monkeypatch.setattr(
        evaluator,
        '_resolve_policy_target',
        lambda _cfg: (object(), '/tmp/checkpoints/openpi/1000', 'pi0_cfg'),
    )
    monkeypatch.setattr(
        evaluator,
        '_is_port_open',
        Mock(return_value=False),
    )
    monkeypatch.setattr(
        evaluator, '_start_policy_server_process', start_mock
    )

    with pytest.raises(RuntimeError, match='unreachable'):
        evaluator._create_policy_client(cfg)

    start_mock.assert_not_called()


def test_eval_vla_arena_terminates_managed_server_process(
    monkeypatch, tmp_path: pathlib.Path
):
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')

    managed_process = Mock()
    managed_process.poll.return_value = None
    managed_process.wait.return_value = 0
    client_obj = Mock()
    log_file = Mock()

    cfg = evaluator.GenerateConfig(
        task_suite_name='safety_static_obstacles',
        result_json_path=str(tmp_path / 'result.json'),
        use_replacements=False,
    )

    monkeypatch.setattr(
        evaluator,
        '_create_policy_client',
        lambda _cfg: (
            client_obj,
            'localhost:8000',
            'pi0_cfg',
            managed_process,
        ),
    )
    monkeypatch.setattr(
        evaluator.benchmark,
        'get_benchmark_dict',
        lambda: {'safety_static_obstacles': lambda: object()},
    )
    monkeypatch.setattr(
        evaluator,
        'setup_logging',
        lambda _cfg: (log_file, str(tmp_path / 'eval.log'), 'run-id'),
    )
    monkeypatch.setattr(
        evaluator,
        'run_task',
        lambda *args, **kwargs: (1, 0, 0, 0, 0, 0, 0, 0),
    )
    monkeypatch.setattr(
        evaluator,
        'load_replacements_dict',
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(evaluator.tqdm, 'tqdm', lambda it: it)

    evaluator.eval_vla_arena(cfg)

    managed_process.terminate.assert_called_once()
    managed_process.wait.assert_called_once()
