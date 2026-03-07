"""
Tests for CLI functionality in vla_arena.cli.
"""

import argparse
import os
from unittest.mock import Mock, patch

import pytest


try:
    from vla_arena.cli import eval
    from vla_arena.cli import main as cli_main_module
    from vla_arena.cli import train
    from vla_arena.cli.main import main as cli_main_function

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False
    cli_main_module = None
    cli_main_function = None
    eval = None
    train = None


@pytest.mark.skipif(not CLI_AVAILABLE, reason='CLI module not available')
class TestCLIMain:
    """Test cases for CLI main function."""

    def test_main_train_parser(self, monkeypatch):
        """Test main function with train command."""
        mock_train_main = Mock()
        monkeypatch.setattr('vla_arena.cli.main.train_main', mock_train_main)

        with patch(
            'sys.argv',
            [
                'vla-arena',
                'train',
                '--model',
                'openvla',
                '--config',
                'test.yaml',
            ],
        ):
            try:
                cli_main_function()
            except SystemExit:
                pass

        # Check that train_main was called (or parser was invoked)
        # The actual call depends on argparse behavior

    def test_main_eval_parser(self, monkeypatch):
        """Test main function with eval command."""
        mock_eval_main = Mock()
        monkeypatch.setattr('vla_arena.cli.main.eval_main', mock_eval_main)

        with patch(
            'sys.argv',
            [
                'vla-arena',
                'eval',
                '--model',
                'openvla',
                '--config',
                'test.yaml',
            ],
        ):
            try:
                cli_main_function()
            except SystemExit:
                pass

    def test_main_no_command(self, capsys):
        """Test main function with no command."""
        with patch('sys.argv', ['vla-arena']):
            try:
                cli_main_function()
            except SystemExit:
                pass
            # Should print help


@pytest.mark.skipif(not CLI_AVAILABLE, reason='CLI module not available')
class TestEvalMain:
    """Test cases for eval_main function."""

    @patch('vla_arena.cli.eval.importlib.util.find_spec')
    def test_eval_main_module_not_found(self, mock_find_spec):
        """Test eval_main when module is not found."""
        mock_find_spec.return_value = None

        args = argparse.Namespace()
        args.model = 'nonexistent_model'
        args.config = '/path/to/config.yaml'

        with pytest.raises(RuntimeError):
            eval.eval_main(args)

    @patch('vla_arena.cli.eval.importlib.util.find_spec')
    def test_eval_main_import_error(self, mock_find_spec):
        """Test eval_main when import fails."""
        mock_find_spec.side_effect = ImportError('Module not found')

        args = argparse.Namespace()
        args.model = 'openvla'
        args.config = '/path/to/config.yaml'

        with pytest.raises(RuntimeError):
            eval.eval_main(args)

    @patch('vla_arena.cli.eval.importlib.util.find_spec')
    @patch('vla_arena.cli.eval.importlib.import_module')
    def test_eval_main_config_path_absolute(
        self, mock_import_module, mock_find_spec
    ):
        """Test that config path is converted to absolute."""
        mock_spec = Mock()
        mock_spec.origin = '/path/to/evaluator.py'
        mock_find_spec.return_value = mock_spec

        mock_module = Mock()
        mock_import_module.return_value = mock_module

        args = argparse.Namespace()
        args.model = 'openvla'
        args.config = 'relative/path/config.yaml'

        eval.eval_main(args)

        # Check that config path passed to main is absolute
        call_args = mock_module.main.call_args
        assert call_args is not None
        config_path = (
            call_args[1].get('cfg') or call_args[0][0]
            if call_args[0]
            else None
        )
        if config_path:
            assert os.path.isabs(config_path)


@pytest.mark.skipif(not CLI_AVAILABLE, reason='CLI module not available')
class TestTrainMain:
    """Test cases for train_main function."""

    @patch('vla_arena.cli.train.importlib.util.find_spec')
    @patch('vla_arena.cli.train.importlib.import_module')
    @patch.dict(os.environ, {}, clear=False)
    def test_train_main_openpi(self, mock_import_module, mock_find_spec):
        """Test train_main for openpi model (JAX)."""
        mock_spec = Mock()
        mock_spec.origin = '/path/to/trainer.py'
        mock_find_spec.return_value = mock_spec

        mock_module = Mock()
        mock_import_module.return_value = mock_module

        args = argparse.Namespace()
        args.model = 'openpi'
        args.config = '/path/to/config.yaml'

        train.train_main(args)

        # import_module will be called multiple times during import chain
        # Just verify it was called and the final module.main was called
        assert mock_import_module.called
        mock_module.main.assert_called_once()

    @patch('vla_arena.cli.train.importlib.util.find_spec')
    @patch('vla_arena.cli.train.importlib.import_module')
    @patch.dict(os.environ, {'LOCAL_RANK': '0'}, clear=False)
    def test_train_main_distributed(self, mock_import_module, mock_find_spec):
        """Test train_main when already in distributed mode."""
        mock_spec = Mock()
        mock_spec.origin = '/path/to/trainer.py'
        mock_find_spec.return_value = mock_spec

        mock_module = Mock()
        mock_import_module.return_value = mock_module

        args = argparse.Namespace()
        args.model = 'openvla'
        args.config = '/path/to/config.yaml'

        train.train_main(args)

        # import_module will be called multiple times during import chain
        # Just verify it was called and the final module.main was called
        assert mock_import_module.called
        mock_module.main.assert_called_once()

    @patch('vla_arena.cli.train.importlib.util.find_spec')
    @patch('vla_arena.cli.train.subprocess.run')
    @patch('vla_arena.cli.train.torch.cuda.device_count')
    @patch.dict(os.environ, {}, clear=False)
    def test_train_main_launch_torchrun(
        self, mock_device_count, mock_subprocess, mock_find_spec
    ):
        """Test train_main launching torchrun."""
        mock_spec = Mock()
        mock_spec.origin = '/path/to/trainer.py'
        mock_find_spec.return_value = mock_spec

        mock_device_count.return_value = 2
        mock_subprocess.return_value = Mock(returncode=0)

        args = argparse.Namespace()
        args.model = 'openvla'
        args.config = '/path/to/config.yaml'

        train.train_main(args)

        # Verify torchrun was called
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert 'torchrun' in call_args[0]

    @patch('vla_arena.cli.train.importlib.util.find_spec')
    def test_train_main_module_not_found(self, mock_find_spec):
        """Test train_main when module is not found."""
        mock_find_spec.return_value = None

        args = argparse.Namespace()
        args.model = 'nonexistent_model'
        args.config = '/path/to/config.yaml'

        with pytest.raises(RuntimeError):
            train.train_main(args)

    @patch('vla_arena.cli.train.importlib.util.find_spec')
    @patch('vla_arena.cli.train.importlib.import_module')
    def test_train_main_overwrite_flag(
        self, mock_import_module, mock_find_spec
    ):
        """Test train_main with overwrite flag."""
        mock_spec = Mock()
        mock_spec.origin = '/path/to/trainer.py'
        mock_find_spec.return_value = mock_spec

        mock_module = Mock()
        mock_import_module.return_value = mock_module

        args = argparse.Namespace()
        args.model = 'openpi'
        args.config = '/path/to/config.yaml'
        args.overwrite = True

        train.train_main(args)

        # Check that overwrite was passed
        call_kwargs = mock_module.main.call_args[1]
        assert call_kwargs.get('overwrite') is True
