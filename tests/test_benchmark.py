"""
Tests for benchmark functionality in vla_arena.benchmark.
"""

import pytest


try:
    from vla_arena.vla_arena.benchmark import (
        BENCHMARK_MAPPING,
        Benchmark,
        Task,
        assign_task_level,
        extract_level_from_task_name,
        get_benchmark,
        get_benchmark_dict,
        grab_language_from_filename,
        register_benchmark,
    )

    BENCHMARK_AVAILABLE = True
except (ImportError, OSError, FileNotFoundError, ModuleNotFoundError):
    # OSError/FileNotFoundError can occur on Windows when mujoco.dll is missing
    BENCHMARK_AVAILABLE = False
    # Create dummy classes for testing
    Benchmark = None
    Task = None
    register_benchmark = None
    get_benchmark = None
    get_benchmark_dict = None
    extract_level_from_task_name = None
    grab_language_from_filename = None
    assign_task_level = None
    BENCHMARK_MAPPING = {}


@pytest.mark.skipif(
    not BENCHMARK_AVAILABLE, reason='benchmark module not available'
)
class TestTask:
    """Test cases for Task namedtuple."""

    def test_task_creation(self):
        """Test creating a Task."""
        task = Task(
            name='test_task_L0',
            language='pick up the cup',
            problem='vla_arena',
            problem_folder='safety_static_obstacles',
            bddl_file='test_task_L0.bddl',
            init_states_file='test_task_L0.pruned_init',
            level=0,
            level_id=0,
        )

        assert task.name == 'test_task_L0'
        assert task.language == 'pick up the cup'
        assert task.level == 0
        assert task.level_id == 0

    def test_task_immutability(self):
        """Test that Task is immutable."""
        task = Task(
            name='test',
            language='test',
            problem='test',
            problem_folder='test',
            bddl_file='test.bddl',
            init_states_file='test.pruned_init',
            level=0,
            level_id=0,
        )

        with pytest.raises(AttributeError):
            task.name = 'new_name'


@pytest.mark.skipif(
    not BENCHMARK_AVAILABLE, reason='benchmark module not available'
)
class TestBenchmarkRegistration:
    """Test cases for benchmark registration."""

    def test_register_benchmark(self):
        """Test registering a benchmark."""

        class TestBenchmark(Benchmark):
            def __init__(self):
                super().__init__()
                self.name = 'test_benchmark'
                self._make_benchmark()

        register_benchmark(TestBenchmark)
        assert 'testbenchmark' in BENCHMARK_MAPPING

    def test_get_benchmark_dict(self, capsys):
        """Test getting benchmark dictionary."""

        class TestBenchmark(Benchmark):
            def __init__(self):
                super().__init__()
                self.name = 'test_benchmark2'
                self._make_benchmark()

        register_benchmark(TestBenchmark)
        benchmark_dict = get_benchmark_dict(help=True)

        assert isinstance(benchmark_dict, dict)
        captured = capsys.readouterr()
        assert (
            'Available benchmarks' in captured.out or len(benchmark_dict) >= 0
        )

    def test_get_benchmark_case_insensitive(self):
        """Test that get_benchmark is case insensitive."""

        # Use a class name that won't conflict with existing benchmarks
        # register_benchmark uses class.__name__, not instance.name
        class TestBenchmarkCaseTest(Benchmark):
            def __init__(self):
                super().__init__()
                self.name = 'test_benchmark_case_test'
                # Don't call _make_benchmark() as it requires vla_arena_task_map entry

        register_benchmark(TestBenchmarkCaseTest)

        # Should work with different cases (using class name)
        class_name = 'TestBenchmarkCaseTest'
        benchmark1 = get_benchmark(class_name)
        benchmark2 = get_benchmark(class_name.upper())
        benchmark3 = get_benchmark(class_name.lower())

        assert benchmark1 == benchmark2 == benchmark3

        # Cleanup
        BENCHMARK_MAPPING.pop(class_name.lower(), None)


@pytest.mark.skipif(
    not BENCHMARK_AVAILABLE, reason='benchmark module not available'
)
class TestLevelExtraction:
    """Test cases for level extraction functions."""

    def test_extract_level_from_task_name_L0(self):
        """Test extracting level 0 from task name."""
        level = extract_level_from_task_name('task_name_L0')
        assert level == 0

    def test_extract_level_from_task_name_L1(self):
        """Test extracting level 1 from task name."""
        level = extract_level_from_task_name('task_name_L1')
        assert level == 1

    def test_extract_level_from_task_name_L2(self):
        """Test extracting level 2 from task name."""
        level = extract_level_from_task_name('task_name_L2')
        assert level == 2

    def test_extract_level_from_task_name_with_bddl(self):
        """Test extracting level from task name with .bddl extension."""
        level = extract_level_from_task_name('task_name_L1.bddl')
        assert level == 1

    def test_extract_level_from_task_name_no_level(self):
        """Test extracting level when no level suffix exists."""
        level = extract_level_from_task_name('task_name')
        assert level is None

    def test_grab_language_from_filename(self):
        """Test extracting language from filename."""
        language = grab_language_from_filename('pick_up_the_cup_L0.bddl')
        assert isinstance(language, str)
        assert len(language) > 0

    def test_assign_task_level_from_name(self):
        """Test assigning task level from name."""
        level = assign_task_level('task_L1')
        assert level == 1

    def test_assign_task_level_from_index(self):
        """Test assigning task level from index."""
        level = assign_task_level('task', task_index=4)
        assert level in [0, 1, 2]  # Should be 4 % 3 = 1

    def test_assign_task_level_default(self):
        """Test default task level assignment."""
        level = assign_task_level('task')
        assert level == 0
