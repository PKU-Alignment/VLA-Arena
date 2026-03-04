from __future__ import annotations

import logging
import os
import pickle
import pathlib
import sys
import types
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


def test_resolve_checkpoint_dir_hf_repo_latest(monkeypatch, tmp_path: pathlib.Path):
    downloaded_repo = tmp_path / 'hf_repo'
    _make_step_dir(downloaded_repo, 3)
    _make_step_dir(downloaded_repo, 20)
    download_mock = Mock(return_value=downloaded_repo)
    monkeypatch.setattr(
        workflow_utils, '_download_hf_model_repo', download_mock
    )

    resolved = workflow_utils.resolve_checkpoint_dir(
        'org/repo', train_cfg=None, policy_checkpoint_step='latest'
    )

    assert pathlib.Path(resolved) == (downloaded_repo / '20').resolve()
    download_mock.assert_called_once_with('org/repo')


def test_resolve_checkpoint_dir_hf_repo_numeric_step(
    monkeypatch, tmp_path: pathlib.Path
):
    downloaded_repo = tmp_path / 'hf_repo'
    _make_step_dir(downloaded_repo, 9)
    _make_step_dir(downloaded_repo, 42)
    monkeypatch.setattr(
        workflow_utils,
        '_download_hf_model_repo',
        Mock(return_value=downloaded_repo),
    )

    resolved = workflow_utils.resolve_checkpoint_dir(
        'org/repo', train_cfg=None, policy_checkpoint_step='42'
    )

    assert pathlib.Path(resolved) == (downloaded_repo / '42').resolve()


def test_resolve_checkpoint_dir_prefers_existing_local_path_over_hf_repo(
    monkeypatch, tmp_path: pathlib.Path
):
    local_repo = tmp_path / 'org' / 'repo'
    _make_step_dir(local_repo, 11)
    monkeypatch.chdir(tmp_path)
    download_mock = Mock(
        side_effect=AssertionError('HF repo should not be downloaded')
    )
    monkeypatch.setattr(
        workflow_utils, '_download_hf_model_repo', download_mock
    )

    resolved = workflow_utils.resolve_checkpoint_dir(
        'org/repo', train_cfg=None, policy_checkpoint_step='latest'
    )

    assert pathlib.Path(resolved) == (local_repo / '11').resolve()
    download_mock.assert_not_called()


def test_resolve_checkpoint_dir_hf_repo_download_failure_has_guidance(
    monkeypatch,
):
    monkeypatch.setattr(
        workflow_utils,
        '_download_hf_model_repo',
        Mock(side_effect=RuntimeError('network down')),
    )

    with pytest.raises(
        FileNotFoundError, match='could not be downloaded from Hugging Face'
    ) as exc_info:
        workflow_utils.resolve_checkpoint_dir(
            'org/repo', train_cfg=None, policy_checkpoint_step='latest'
        )

    assert 'prefix it with ./' in str(exc_info.value)


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


def test_load_train_config_from_yaml_fallbacks_to_packaged_reference(
    monkeypatch, tmp_path: pathlib.Path
):
    yaml_path = tmp_path / 'openpi.yaml'
    yaml_path.write_text(
        'name: "pi0_vla_arena_low_mem_finetune"\n'
        'exp_name: "openpi_test"\n',
        encoding='utf-8',
    )

    fake_config_module = SimpleNamespace(cli=Mock(return_value='cfg_obj'))
    import_module_mock = Mock(return_value=fake_config_module)
    monkeypatch.setattr(
        workflow_utils.importlib, 'import_module', import_module_mock
    )

    resolver_mock = Mock(return_value=yaml_path)
    monkeypatch.setattr(
        workflow_utils, 'resolve_packaged_config_reference', resolver_mock
    )

    cfg = workflow_utils.load_train_config_from_yaml(
        'vla_arena/configs/train/_pypi_fallback_test.yaml'
    )

    assert cfg == 'cfg_obj'
    resolver_mock.assert_called_once()
    import_module_mock.assert_called_once_with(
        'vla_arena.models.openpi.src.openpi.training.config'
    )
    fake_config_module.cli.assert_called_once()


def _mock_openpi_config_cli(monkeypatch):
    captured = {}

    def _cli():
        captured['argv'] = workflow_utils.sys.argv.copy()
        return 'cfg_obj'

    fake_config_module = SimpleNamespace(cli=Mock(side_effect=_cli))
    import_module_mock = Mock(return_value=fake_config_module)
    monkeypatch.setattr(
        workflow_utils.importlib, 'import_module', import_module_mock
    )
    return captured


def test_load_train_config_from_yaml_local_repo_path_sets_env_and_maps_repo_id(
    monkeypatch, tmp_path: pathlib.Path
):
    local_repo = tmp_path / 'datasets' / 'mysets' / 'vla_openpi'
    local_repo.mkdir(parents=True)
    yaml_path = tmp_path / 'openpi_local.yaml'
    yaml_path.write_text(
        'name: "pi0_vla_arena_low_mem_finetune"\n'
        'exp_name: "openpi_local_test"\n'
        f'data:\n  repo_id: "{local_repo}"\n',
        encoding='utf-8',
    )

    monkeypatch.delenv('HF_LEROBOT_HOME', raising=False)
    captured = _mock_openpi_config_cli(monkeypatch)

    cfg = workflow_utils.load_train_config_from_yaml(yaml_path)

    assert cfg == 'cfg_obj'
    assert os.getenv('HF_LEROBOT_HOME') == str(tmp_path / 'datasets')
    assert '--data.repo_id=mysets/vla_openpi' in captured['argv']


def test_load_train_config_from_yaml_local_repo_path_overrides_existing_hf_home(
    monkeypatch, tmp_path: pathlib.Path, caplog
):
    local_repo = tmp_path / 'datasets' / 'mysets' / 'vla_openpi'
    local_repo.mkdir(parents=True)
    yaml_path = tmp_path / 'openpi_local_override.yaml'
    yaml_path.write_text(
        'name: "pi0_vla_arena_low_mem_finetune"\n'
        'exp_name: "openpi_local_override"\n'
        f'data:\n  repo_id: "{local_repo}"\n',
        encoding='utf-8',
    )

    monkeypatch.setenv('HF_LEROBOT_HOME', '/tmp/old_hf_home')
    captured = _mock_openpi_config_cli(monkeypatch)
    caplog.set_level(logging.WARNING)

    workflow_utils.load_train_config_from_yaml(yaml_path)

    assert os.getenv('HF_LEROBOT_HOME') == str(tmp_path / 'datasets')
    assert '--data.repo_id=mysets/vla_openpi' in captured['argv']
    assert any(
        'Overriding HF_LEROBOT_HOME' in record.message
        for record in caplog.records
    )


def test_load_train_config_from_yaml_nonexistent_repo_id_keeps_hf_behavior(
    monkeypatch, tmp_path: pathlib.Path
):
    yaml_path = tmp_path / 'openpi_hf.yaml'
    yaml_path.write_text(
        'name: "pi0_vla_arena_low_mem_finetune"\n'
        'exp_name: "openpi_hf_test"\n'
        'data:\n  repo_id: "org/repo"\n',
        encoding='utf-8',
    )

    monkeypatch.setenv('HF_LEROBOT_HOME', '/tmp/keep_this_home')
    captured = _mock_openpi_config_cli(monkeypatch)

    workflow_utils.load_train_config_from_yaml(yaml_path)

    assert os.getenv('HF_LEROBOT_HOME') == '/tmp/keep_this_home'
    assert '--data.repo_id=org/repo' in captured['argv']


def test_local_repo_mapping_fallback_single_level(
    monkeypatch, tmp_path: pathlib.Path
):
    local_repo = tmp_path / 'dataset_only'
    local_repo.mkdir(parents=True)
    yaml_path = tmp_path / 'openpi_local_single_level.yaml'
    yaml_path.write_text(
        'name: "pi0_vla_arena_low_mem_finetune"\n'
        'exp_name: "openpi_local_single"\n'
        'data:\n  repo_id: "dataset_only"\n',
        encoding='utf-8',
    )

    monkeypatch.delenv('HF_LEROBOT_HOME', raising=False)
    monkeypatch.chdir(tmp_path)
    captured = _mock_openpi_config_cli(monkeypatch)

    workflow_utils.load_train_config_from_yaml(yaml_path)

    assert os.getenv('HF_LEROBOT_HOME') == str(tmp_path)
    assert '--data.repo_id=dataset_only' in captured['argv']


def _mock_openpi_training_config_module(
    monkeypatch, cfg_name: str = 'pi0_vla_arena_low_mem_finetune'
):
    module_name = 'vla_arena.models.openpi.src.openpi.training.config'
    fake_module = types.ModuleType(module_name)
    get_config = Mock(return_value=SimpleNamespace(name=cfg_name))
    fake_module.get_config = get_config
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    return get_config


def test_resolve_policy_target_prefers_train_config_name(monkeypatch, caplog):
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')
    get_config = _mock_openpi_training_config_module(monkeypatch)
    load_train_mock = Mock(
        side_effect=AssertionError('legacy train_config_path should be ignored')
    )
    monkeypatch.setattr(
        evaluator, 'load_train_config_from_yaml', load_train_mock
    )
    resolve_checkpoint_mock = Mock(return_value='/tmp/openpi/1000')
    monkeypatch.setattr(
        evaluator, 'resolve_checkpoint_dir', resolve_checkpoint_mock
    )
    caplog.set_level(logging.WARNING)

    cfg = evaluator.GenerateConfig(
        train_config_name='pi0_vla_arena_low_mem_finetune',
        train_config_path='vla_arena/configs/train/openpi.yaml',
        policy_config_name='legacy_alias',
        policy_checkpoint_dir='org/repo',
    )
    train_cfg, checkpoint_dir, config_name = evaluator._resolve_policy_target(
        cfg
    )

    assert checkpoint_dir == '/tmp/openpi/1000'
    assert config_name == 'pi0_vla_arena_low_mem_finetune'
    assert train_cfg.name == 'pi0_vla_arena_low_mem_finetune'
    get_config.assert_called_once_with('pi0_vla_arena_low_mem_finetune')
    resolve_checkpoint_mock.assert_called_once_with(
        'org/repo',
        train_cfg=None,
        policy_checkpoint_step='latest',
    )
    assert any(
        'train_config_path is deprecated and ignored' in record.message
        for record in caplog.records
    )
    assert any(
        'policy_config_name is deprecated and ignored' in record.message
        for record in caplog.records
    )


def test_resolve_policy_target_requires_checkpoint_dir_with_train_config_name(
    monkeypatch,
):
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')
    _mock_openpi_training_config_module(monkeypatch)

    cfg = evaluator.GenerateConfig(
        train_config_name='pi0_vla_arena_low_mem_finetune',
        policy_checkpoint_dir=None,
    )
    with pytest.raises(
        ValueError, match='policy_checkpoint_dir must be set'
    ):
        evaluator._resolve_policy_target(cfg)


def test_resolve_policy_target_train_config_path_legacy_warns(
    monkeypatch, caplog
):
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')
    legacy_cfg = SimpleNamespace(name='legacy_train_cfg')
    load_train_mock = Mock(return_value=legacy_cfg)
    monkeypatch.setattr(
        evaluator, 'load_train_config_from_yaml', load_train_mock
    )
    resolve_checkpoint_mock = Mock(return_value='/tmp/openpi/2000')
    monkeypatch.setattr(
        evaluator, 'resolve_checkpoint_dir', resolve_checkpoint_mock
    )
    caplog.set_level(logging.WARNING)

    cfg = evaluator.GenerateConfig(
        train_config_path='vla_arena/configs/train/openpi.yaml'
    )
    train_cfg, checkpoint_dir, config_name = evaluator._resolve_policy_target(
        cfg
    )

    assert train_cfg is legacy_cfg
    assert checkpoint_dir == '/tmp/openpi/2000'
    assert config_name == 'legacy_train_cfg'
    load_train_mock.assert_called_once_with(
        'vla_arena/configs/train/openpi.yaml'
    )
    resolve_checkpoint_mock.assert_called_once_with(
        None,
        legacy_cfg,
        'latest',
    )
    assert any(
        'train_config_path is deprecated' in record.message
        for record in caplog.records
    )


def test_resolve_policy_target_policy_config_name_legacy_warns(
    monkeypatch, caplog
):
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')
    get_config = _mock_openpi_training_config_module(monkeypatch, 'legacy_cfg')
    resolve_checkpoint_mock = Mock(return_value='/tmp/openpi/3000')
    monkeypatch.setattr(
        evaluator, 'resolve_checkpoint_dir', resolve_checkpoint_mock
    )
    caplog.set_level(logging.WARNING)

    cfg = evaluator.GenerateConfig(
        policy_config_name='legacy_cfg',
        policy_checkpoint_dir='/tmp/checkpoints',
    )
    train_cfg, checkpoint_dir, config_name = evaluator._resolve_policy_target(
        cfg
    )

    assert train_cfg.name == 'legacy_cfg'
    assert checkpoint_dir == '/tmp/openpi/3000'
    assert config_name == 'legacy_cfg'
    get_config.assert_called_once_with('legacy_cfg')
    resolve_checkpoint_mock.assert_called_once_with(
        '/tmp/checkpoints',
        train_cfg,
        'latest',
    )
    assert any(
        'policy_config_name is deprecated' in record.message
        for record in caplog.records
    )


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


def test_build_serve_policy_command_places_port_before_subcommand():
    evaluator = pytest.importorskip('vla_arena.models.openpi.evaluator')
    cfg = evaluator.GenerateConfig(port=8001)

    cmd = evaluator._build_serve_policy_command(
        cfg,
        config_name='pi0_vla_arena_low_mem_finetune',
        checkpoint_dir='/tmp/openpi/1000',
    )

    assert '--port' in cmd
    assert 'policy:checkpoint' in cmd
    assert cmd.index('--port') < cmd.index('policy:checkpoint')
    assert cmd[cmd.index('--port') + 1] == '8001'
    assert '--policy.config' in cmd
    assert '--policy.dir' in cmd


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
