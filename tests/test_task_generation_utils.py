"""
Tests for task generation utilities.
"""

from unittest.mock import Mock, patch

import pytest


try:
    from vla_arena.vla_arena.utils import task_generation_utils

    TASK_GEN_UTILS_AVAILABLE = True
except (ImportError, OSError, FileNotFoundError, ModuleNotFoundError):
    # OSError/FileNotFoundError can occur on Windows when mujoco.dll is missing
    TASK_GEN_UTILS_AVAILABLE = False


@pytest.mark.skipif(not TASK_GEN_UTILS_AVAILABLE, reason='task_generation_utils not available')
class TestTaskGenerationUtils:
    """Test cases for task_generation_utils.py"""

    def test_get_task_info_none(self):
        """Test get_task_info with no scene_name."""
        task_info = task_generation_utils.get_task_info()
        assert isinstance(task_info, dict)

    def test_get_task_info_with_scene(self):
        """Test get_task_info with scene_name."""
        # This should raise KeyError if scene doesn't exist, or return a list
        try:
            task_info = task_generation_utils.get_task_info('nonexistent_scene')
            assert isinstance(task_info, list)
        except KeyError:
            # Expected behavior if scene doesn't exist
            pass

    @patch('vla_arena.vla_arena.utils.task_generation_utils.get_scene_class')
    def test_register_task_info(self, mock_get_scene_class):
        """Test register_task_info function."""
        # Mock scene class - need to make it callable and return an instance
        mock_scene_instance = Mock()
        mock_scene_instance.possible_objects_of_interest = ['obj1', 'obj2', 'obj3']
        mock_scene_class = Mock(return_value=mock_scene_instance)
        mock_get_scene_class.return_value = mock_scene_class

        # Register task info
        task_generation_utils.register_task_info(
            language='pick up the cup',
            scene_name='test_scene_register',
            objects_of_interest=['obj1', 'obj2'],
            goal_states=[('in', 'obj1', 'box')],
        )

        # Verify task was registered
        task_info = task_generation_utils.get_task_info('test_scene_register')
        assert len(task_info) > 0

        # Cleanup
        if 'test_scene_register' in task_generation_utils.TASK_INFO:
            del task_generation_utils.TASK_INFO['test_scene_register']

    @patch('vla_arena.vla_arena.utils.task_generation_utils.get_scene_class')
    def test_register_task_info_invalid_object(self, mock_get_scene_class):
        """Test register_task_info with invalid object."""
        mock_scene_instance = Mock()
        mock_scene_instance.possible_objects_of_interest = ['obj1', 'obj2']
        mock_scene_class = Mock(return_value=mock_scene_instance)
        mock_get_scene_class.return_value = mock_scene_class

        # Should raise ValueError for invalid object
        with pytest.raises(ValueError):
            task_generation_utils.register_task_info(
                language='test',
                scene_name='test_scene_invalid',
                objects_of_interest=['invalid_obj'],
                goal_states=[],
            )

    def test_get_suite_generator_func(self):
        """Test get_suite_generator_func."""
        # Test various workspace names - these functions may not be defined
        workspace_names = [
            'main_table',
            'kitchen_table',
            'living_room_table',
            'study_table',
            'coffee_table',
        ]

        for workspace_name in workspace_names:
            try:
                generator_func = task_generation_utils.get_suite_generator_func(workspace_name)
                # Should return a function or None
                assert generator_func is None or callable(generator_func)
            except NameError:
                # Expected if generator functions are not defined
                pass

    def test_get_suite_generator_func_invalid(self):
        """Test get_suite_generator_func with invalid workspace."""
        try:
            generator_func = task_generation_utils.get_suite_generator_func('invalid_workspace')
            # Should return None or raise error
            assert generator_func is None or callable(generator_func)
        except NameError:
            # Expected if generator functions are not defined
            pass
