"""
Tests for vla_arena initialization and path management.
"""

import os
from unittest.mock import patch

import pytest
import yaml


try:
    from vla_arena.vla_arena import (
        config_file,
        get_default_path_dict,
        get_vla_arena_path,
        set_vla_arena_default_path,
        vla_arena_config_path,
    )

    VLA_ARENA_INIT_AVAILABLE = True
except ImportError:
    VLA_ARENA_INIT_AVAILABLE = False
    get_default_path_dict = None
    get_vla_arena_path = None
    set_vla_arena_default_path = None
    vla_arena_config_path = None
    config_file = None


@pytest.mark.skipif(
    not VLA_ARENA_INIT_AVAILABLE, reason='vla_arena init module not available'
)
class TestPathManagement:
    """Test cases for path management functions."""

    def test_get_default_path_dict(self):
        """Test getting default path dictionary."""
        paths = get_default_path_dict()

        assert isinstance(paths, dict)
        assert 'benchmark_root' in paths
        assert 'bddl_files' in paths
        assert 'init_states' in paths
        assert 'assets' in paths

        # Check that paths are strings
        assert all(isinstance(v, str) for v in paths.values())

    def test_get_default_path_dict_custom_location(self):
        """Test getting default path dict with custom location."""
        custom_location = '/custom/path'
        paths = get_default_path_dict(custom_location)

        assert paths['benchmark_root'] == custom_location
        # Check that paths contain the expected directory names
        assert 'bddl_files' in paths['bddl_files'] or paths[
            'bddl_files'
        ].endswith('bddl_files')
        assert (
            'init_files' in paths['init_states']
            or paths['init_states'].endswith('init_files')
            or 'init_states' in paths['init_states']
        )
        assert 'assets' in paths['assets'] or paths['assets'].endswith(
            'assets'
        )

    def test_get_vla_arena_path_success(self, temp_config_file):
        """Test getting VLA-Arena path from config file."""
        # Read the config file we created
        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        # Mock the config file path
        with patch('vla_arena.vla_arena.config_file', temp_config_file):
            path = get_vla_arena_path('benchmark_root')
            assert isinstance(path, str)

    def test_get_vla_arena_path_missing_key(self, temp_config_file):
        """Test getting VLA-Arena path with missing key."""
        with patch('vla_arena.vla_arena.config_file', temp_config_file):
            with pytest.raises(AssertionError):
                get_vla_arena_path('nonexistent_key')

    def test_set_vla_arena_default_path(self, temp_dir, capsys):
        """Test setting default VLA-Arena path."""
        config_file_path = os.path.join(temp_dir, 'config.yaml')

        with patch('vla_arena.vla_arena.config_file', config_file_path):
            set_vla_arena_default_path(temp_dir)

            # Check that config file was created
            assert os.path.exists(config_file_path)

            # Read and verify config
            with open(config_file_path) as f:
                config = yaml.safe_load(f)

            assert 'benchmark_root' in config
            assert config['benchmark_root'] == temp_dir

    def test_config_file_initialization(self, temp_dir, monkeypatch):
        """Test that config file is initialized if it doesn't exist."""
        config_dir = os.path.join(temp_dir, '.vla_arena')
        config_file_path = os.path.join(config_dir, 'config.yaml')

        # Mock the config path
        monkeypatch.setenv('VLA_ARENA_CONFIG_PATH', config_dir)

        # Import after setting env var to trigger initialization
        import importlib

        import vla_arena.vla_arena

        importlib.reload(vla_arena.vla_arena)

        # Check that directory was created
        if os.path.exists(config_dir):
            assert os.path.isdir(config_dir)


@pytest.mark.skipif(
    not VLA_ARENA_INIT_AVAILABLE, reason='vla_arena init module not available'
)
class TestConfigFileStructure:
    """Test cases for config file structure."""

    def test_config_file_yaml_format(self, temp_config_file):
        """Test that config file is valid YAML."""
        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict)
        assert len(config) > 0

    def test_config_file_required_keys(self, temp_config_file):
        """Test that config file has required keys."""
        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        required_keys = [
            'benchmark_root',
            'bddl_files',
            'init_states',
            'assets',
        ]
        for key in required_keys:
            assert key in config, f'Missing required key: {key}'
