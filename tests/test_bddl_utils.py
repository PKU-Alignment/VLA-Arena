"""
Tests for BDDL generation utilities.
"""

import os
from unittest.mock import Mock, patch

import pytest


try:
    from vla_arena.vla_arena.utils import bddl_generation_utils

    BDDL_UTILS_AVAILABLE = True
except ImportError:
    BDDL_UTILS_AVAILABLE = False


@pytest.mark.skipif(not BDDL_UTILS_AVAILABLE, reason='bddl_generation_utils not available')
class TestBDDLGenerationUtils:
    """Test cases for bddl_generation_utils.py"""

    def test_print_result(self, capsys):
        """Test print_result function."""
        result = ['line1', 'line2', 'line3']
        bddl_generation_utils.print_result(result)

        captured = capsys.readouterr()
        assert 'line1' in captured.out
        assert 'line2' in captured.out
        assert 'line3' in captured.out

    def test_get_result(self):
        """Test get_result function."""
        result = ['line1', 'line2', 'line3']
        output = bddl_generation_utils.get_result(result)

        assert isinstance(output, str)
        assert 'line1' in output
        assert 'line2' in output
        assert 'line3' in output

    def test_save_to_file(self, temp_dir):
        """Test save_to_file function."""
        # save_to_file expects a string result (from get_result), not a list
        result_list = ['(define (problem test)', '(:domain robosuite)', ')']
        result = bddl_generation_utils.get_result(result_list)  # Convert to string
        scene_name = 'TEST_SCENE'
        language = 'pick up the cup'

        file_path = bddl_generation_utils.save_to_file(
            result, scene_name, language, folder=temp_dir
        )

        assert os.path.exists(file_path)
        assert scene_name.upper() in file_path
        assert file_path.endswith('.bddl')

        # Check file contents
        with open(file_path) as f:
            content = f.read()
            assert 'define' in content.lower()

    def test_pddl_definition_decorator(self):
        """Test PDDLDefinition decorator."""

        @bddl_generation_utils.PDDLDefinition(problem_name='test_problem')
        def test_problem():
            return ['(:objects obj1 - object)', '(:init)']

        result = test_problem()

        assert isinstance(result, list)
        assert any('test_problem' in line for line in result)
        assert any('robosuite' in line.lower() for line in result)
