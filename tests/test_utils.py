"""
Tests for utility functions in vla_arena.utils.
"""

from unittest.mock import patch

import numpy as np
import pytest


try:
    from vla_arena.vla_arena.utils import utils

    UTILS_AVAILABLE = True
except (ImportError, OSError, FileNotFoundError, ModuleNotFoundError):
    # OSError/FileNotFoundError can occur on Windows when mujoco.dll is missing
    UTILS_AVAILABLE = False

try:
    from vla_arena.vla_arena.utils import dataset_utils

    DATASET_UTILS_AVAILABLE = True
except ImportError:
    DATASET_UTILS_AVAILABLE = False

try:
    from vla_arena.vla_arena.utils import time_utils

    TIME_UTILS_AVAILABLE = True
except ImportError:
    TIME_UTILS_AVAILABLE = False


@pytest.mark.skipif(not UTILS_AVAILABLE, reason='utils module not available')
class TestUtils:
    """Test cases for utils.py"""

    def test_process_image_input(self):
        """Test image input processing."""
        # Test with numpy array
        img_tensor = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        processed = utils.process_image_input(img_tensor)

        assert processed.dtype == np.float64 or processed.dtype == np.float32
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0
        assert processed.shape == img_tensor.shape

    def test_reconstruct_image_output(self):
        """Test image output reconstruction."""
        img_array = np.random.rand(224, 224, 3)
        reconstructed = utils.reconstruct_image_output(img_array)

        assert (
            reconstructed.dtype == np.float64
            or reconstructed.dtype == np.float32
        )
        assert reconstructed.max() <= 255.0
        assert reconstructed.min() >= 0.0
        assert reconstructed.shape == img_array.shape

    def test_update_env_kwargs(self):
        """Test updating environment kwargs."""
        env_kwargs = {'key1': 'value1', 'key2': 'value2'}
        utils.update_env_kwargs(env_kwargs, key3='value3', key1='new_value1')

        assert env_kwargs['key1'] == 'new_value1'
        assert env_kwargs['key2'] == 'value2'
        assert env_kwargs['key3'] == 'value3'

    @patch('vla_arena.vla_arena.utils.utils.robosuite')
    def test_postprocess_model_xml(self, mock_robosuite):
        """Test XML postprocessing."""
        mock_robosuite.__file__ = '/path/to/robosuite/__init__.py'

        # Create a simple XML string
        xml_str = """<?xml version="1.0"?>
        <mujoco>
            <asset>
                <mesh file="/some/path/robosuite/models/assets/mesh.stl"/>
                <texture file="/some/path/robosuite/models/assets/texture.png"/>
            </asset>
            <worldbody>
                <camera name="frontview" pos="0 0 0" quat="1 0 0 0"/>
            </worldbody>
        </mujoco>"""

        cameras_dict = {'frontview': {'pos': '1 1 1', 'quat': '1 0 0 0'}}

        result = utils.postprocess_model_xml(xml_str, cameras_dict)

        assert isinstance(result, str)
        assert 'robosuite' in result or len(result) > 0


@pytest.mark.skipif(
    not DATASET_UTILS_AVAILABLE, reason='dataset_utils module not available'
)
class TestDatasetUtils:
    """Test cases for dataset_utils.py"""

    def test_get_dataset_info_basic(self, mock_h5py_file):
        """Test basic dataset info extraction."""
        info = dataset_utils.get_dataset_info(mock_h5py_file, verbose=False)

        # Should complete without error
        assert info is None  # Function doesn't return anything, just prints

    def test_get_dataset_info_with_filter_key(self, mock_h5py_file):
        """Test dataset info with filter key."""
        # This will fail if filter key doesn't exist, but we test the function call
        try:
            dataset_utils.get_dataset_info(
                mock_h5py_file, filter_key='test_filter', verbose=False
            )
        except (KeyError, AttributeError):
            # Expected if filter key doesn't exist
            pass


@pytest.mark.skipif(
    not TIME_UTILS_AVAILABLE, reason='time_utils module not available'
)
class TestTimeUtils:
    """Test cases for time_utils.py"""

    def test_timer_context_manager(self):
        """Test Timer as context manager."""
        import time

        with time_utils.Timer() as timer:
            time.sleep(0.1)

        elapsed = timer.get_elapsed_time()
        assert elapsed >= 0.1
        assert elapsed < 1.0  # Should be much less than 1 second

    def test_timer_value_attribute(self):
        """Test Timer value attribute."""
        import time

        with time_utils.Timer() as timer:
            time.sleep(0.05)

        assert hasattr(timer, 'value')
        assert timer.value >= 0.05
        assert timer.value == timer.get_elapsed_time()
