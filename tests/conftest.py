"""
Pytest configuration and fixtures for VLA-Arena tests.
"""

import os
import shutil
import tempfile
from unittest.mock import Mock

import pytest
import yaml


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_config_file(temp_dir):
    """Create a temporary config YAML file."""
    config_path = os.path.join(temp_dir, 'config.yaml')
    config_data = {
        'benchmark_root': temp_dir,
        'bddl_files': os.path.join(temp_dir, 'bddl_files'),
        'init_states': os.path.join(temp_dir, 'init_states'),
        'assets': os.path.join(temp_dir, 'assets'),
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def mock_vla_arena_paths(monkeypatch, temp_dir):
    """Mock VLA-Arena path functions to use temporary directory."""

    def mock_get_vla_arena_path(key):
        paths = {
            'benchmark_root': temp_dir,
            'bddl_files': os.path.join(temp_dir, 'bddl_files'),
            'init_states': os.path.join(temp_dir, 'init_states'),
            'assets': os.path.join(temp_dir, 'assets'),
        }
        return paths.get(key, temp_dir)

    monkeypatch.setattr(
        'vla_arena.vla_arena.benchmark.__init__.get_vla_arena_path',
        mock_get_vla_arena_path,
    )
    return temp_dir


@pytest.fixture
def sample_bddl_content():
    """Sample BDDL file content for testing."""
    return """
(define (problem test_problem)
    (:domain robosuite)
    (:requirements)
    (:objects obj1 obj2 - object)
    (:language pick up the red cup)
    (:init
        (on obj1 table)
    )
    (:goal
        (in obj1 box)
    )
)
"""


@pytest.fixture
def sample_task():
    """Create a sample Task namedtuple for testing."""
    from vla_arena.vla_arena.benchmark import Task

    return Task(
        name='test_task_L0',
        language='pick up the red cup',
        problem='vla_arena',
        problem_folder='safety_static_obstacles',
        bddl_file='test_task_L0.bddl',
        init_states_file='test_task_L0.pruned_init',
        level=0,
        level_id=0,
    )


@pytest.fixture
def mock_benchmark():
    """Create a mock benchmark instance."""
    from unittest.mock import Mock

    benchmark = Mock()
    benchmark.name = 'test_benchmark'
    benchmark.n_tasks = 5
    benchmark.tasks = []
    benchmark.level_task_maps = {0: [], 1: [], 2: []}
    return benchmark


@pytest.fixture(autouse=True)
def reset_benchmark_registry():
    """Reset benchmark registry before each test."""
    try:
        from vla_arena.vla_arena.benchmark import BENCHMARK_MAPPING

        original_mapping = BENCHMARK_MAPPING.copy()
        BENCHMARK_MAPPING.clear()
        yield
        BENCHMARK_MAPPING.clear()
        BENCHMARK_MAPPING.update(original_mapping)
    except (ImportError, RuntimeError) as e:
        # Skip if robosuite or other dependencies are not available
        pytest.skip(f'Skipping benchmark registry reset: {e}')
        yield


@pytest.fixture
def mock_env():
    """Create a mock environment for testing."""
    env = Mock()
    env.reset.return_value = {'image': Mock(), 'state': Mock()}
    env.step.return_value = (
        {'image': Mock(), 'state': Mock()},
        0.0,
        False,
        {},
    )
    env.render.return_value = Mock()
    return env


@pytest.fixture
def mock_h5py_file():
    """Create a mock h5py file for dataset testing."""
    import h5py
    import numpy as np

    # Create a temporary h5py file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.hdf5')
    temp_file.close()

    with h5py.File(temp_file.name, 'w') as f:
        # Create sample data structure
        demo_group = f.create_group('data')
        demo_group.attrs['problem_info'] = (
            '{"language_instruction": ["pick up the cup"]}'
        )
        demo_group.attrs['env_args'] = '{"env_name": "test_env"}'

        # Create a sample episode
        episode = demo_group.create_group('demo_0')
        episode.attrs['num_samples'] = 10
        episode.create_dataset('actions', data=np.random.randn(10, 7))
        episode.create_group('obs')
        episode.create_group('next_obs')

    yield temp_file.name

    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)
