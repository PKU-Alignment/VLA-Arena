import abc
import os
import torch
import re

from typing import List, NamedTuple, Optional
from vla_arena.vla_arena import get_vla_arena_path
from vla_arena.vla_arena.benchmark.vla_arena_suite_task_map import vla_arena_task_map
from vla_arena.vla_arena.envs.bddl_utils import *

BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str
    level: int
    level_id: int  # Index within the level


def extract_level_from_task_name(task_name):
    """
    Extract level from task name.
    Assumes task names end with _L0, _L1, or _L2 to indicate level.
    If no level suffix found, returns None.
    """
    # Check for _L0, _L1, _L2 pattern at the end of the task name
    match = re.search(r'_L([0-2])$', task_name)
    if match:
        return int(match.group(1))
    
    # Also check without .bddl extension if present
    task_without_ext = task_name.replace('.bddl', '')
    match = re.search(r'_L([0-2])$', task_without_ext)
    if match:
        return int(match.group(1))
    
    return None


def grab_language_from_filename(x):
    # Remove the level suffix (_L0, _L1, _L2) before processing
    x_clean = re.sub(r'_L[0-2]', '', x)
    
    if x_clean[0].isupper():  # vla_arena-100
        if "SCENE10" in x_clean:
            language = " ".join(x_clean[x_clean.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(x_clean[x_clean.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x_clean.split("_"))
    en = language.find(".bddl")
    return language[:en]

def grab_language_from_bddl_file(bddl_filename, problem_folder, level_dir):
    domain_name = "robosuite"
    bddl_file_path = os.path.join(get_vla_arena_path("bddl_files"), problem_folder, level_dir, bddl_filename)
    tokens = scan_tokens(filename=bddl_file_path)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        objects = {}
        obj_of_interest = []
        initial_state = []
        goal_state = []
        fixtures = {}
        regions = {}
        image_settings={}
        scene_properties = {}
        language_instruction = ""
        cost_state = []
        moving_objects = []
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                if domain_name != group[-1]:
                    raise Exception("Different domain specified in problem file")
            elif t == ":requirements":
                pass
            elif t == ":objects":
                group.pop(0)
                object_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        objects[group.pop(0)] = object_list
                        object_list = []
                    else:
                        object_list.append(group.pop(0))
                if object_list:
                    if not "object" in objects:
                        objects["object"] = []
                    objects["object"] += object_list
            elif t == ":obj_of_interest":
                group.pop(0)
                while group:
                    obj_of_interest.append(group.pop(0))
            elif t == ":fixtures":
                group.pop(0)
                fixture_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        fixtures[group.pop(0)] = fixture_list
                        fixture_list = []
                    else:
                        fixture_list.append(group.pop(0))
                if fixture_list:
                    if not "fixture" in fixtures:
                        fixtures["fixture"] = []
                    fixtures["fixture"] += fixture_list
            elif t == ":regions":
                get_regions(t, regions, group)
            elif t == ":scene_properties":
                get_scenes(t, scene_properties, group)
            elif t == ":language":
                group.pop(0)
                language_instruction = ' '.join(group)
                return language_instruction
            elif t == ":init":
                group.pop(0)
                initial_state = group
            elif t == ":goal":
                package_predicates(group[1], goal_state, "", "goals")
            elif t == ":cost":
                package_predicates(group[1], cost_state, "", "costs")
            elif t == ":moving_objects":
                get_moving_objects(t, moving_objects, group)
            elif t == ":image_settings":
                group.pop(0)
                while group:
                    if group[0].isalpha():
                        image_settings[group.pop(0)]=float(group.pop(1))
            else:
                print(bddl_filename)
                print("%s is not recognized in problem" % t)
        return {
            "problem_name": problem_name,
            "fixtures": fixtures,
            "regions": regions,
            "objects": objects,
            "scene_properties": scene_properties,
            "initial_state": initial_state,
            "goal_state": goal_state,
            "language_instruction": language_instruction,
            "obj_of_interest": obj_of_interest,
            "cost_state": cost_state,
            "moving_objects": moving_objects,
            "image_settings": image_settings,
        }
    else:
        raise Exception(
            f"Problem does not match problem pattern"
        )

def assign_task_level(task_name, task_index=None):
    """
    Assign a level (0, 1, or 2) to a task.
    First tries to extract from task name, then falls back to other strategies.
    """
    # First, try to extract level from task name
    level = extract_level_from_task_name(task_name)
    if level is not None:
        return level
    
    # Fallback: Simple cyclic assignment based on task index
    if task_index is not None:
        return task_index % 3
    
    # Default to level 0 if no other method works
    return 0


# Complete list of all VLA Arena suites
# Organized by category for better readability
vla_arena_suites = [
    # Safety benchmarks
    "safety_dynamic_obstacles",
    "safety_hazard_avoidance", 
    "safety_object_state_preservation",
    "safety_risk_aware_grasping",
    "safety_static_obstacles",
    
    # Robustness benchmarks
    "robustness_dynamic_distractors",
    "robustness_static_distractors",
    "robustness_visual_variations",
    
    # Generalization benchmarks
    "generalization_language_variations",
    "generalization_object_preposition_combinations",
    "generalization_task_workflows",
    "generalization_unseen_objects",
    
    # Other benchmarks
    "long_horizon",

    # Libero benchmarks
    "libero_10",
    "libero_90",
    "libero_spatial",
    "libero_object",
    "libero_goal",
]

# Map suite names to problem folders
# Organized by category for better readability
suite_to_problem_folder = {
    # Safety benchmarks
    "safety_dynamic_obstacles": "safety_dynamic_obstacles",
    "safety_hazard_avoidance": "safety_hazard_avoidance",
    "safety_object_state_preservation": "safety_object_state_preservation",
    "safety_risk_aware_grasping": "safety_risk_aware_grasping",
    "safety_static_obstacles": "safety_static_obstacles",
    
    # Robustness benchmarks
    "robustness_dynamic_distractors": "robustness_dynamic_distractors",
    "robustness_static_distractors": "robustness_static_distractors",
    "robustness_visual_variations": "robustness_visual_variations",
    
    # Generalization benchmarks
    "generalization_language_variations": "generalization_language_variations",
    "generalization_object_preposition_combinations": "generalization_object_preposition_combinations",
    "generalization_task_workflows": "generalization_task_workflows",
    "generalization_unseen_objects": "generalization_unseen_objects",
    
    # Other benchmarks
    "long_horizon": "long_horizon",

    # Libero benchmarks
    "libero_10": "libero_10",
    "libero_90": "libero_90",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
}

task_maps = {}

for vla_arena_suite in vla_arena_suites:
    task_maps[vla_arena_suite] = {0: {}, 1: {}, 2: {}}
    
    # Build task maps using the level-based structure from vla_arena_task_map
    for level in [0, 1, 2]:
        if level in vla_arena_task_map[vla_arena_suite]:
            level_tasks = vla_arena_task_map[vla_arena_suite][level]
            
            for level_id, task in enumerate(level_tasks):
                
                # Determine the actual problem folder name
                problem_folder = suite_to_problem_folder.get(vla_arena_suite, vla_arena_suite)
                level_dir = f"level_{level}"

                # Get language (removing level suffix for processing)
                language = grab_language_from_bddl_file(task + ".bddl", problem_folder, level_dir)

                bddl_filename = f"{task}.bddl"
                init_states_filename = f"{task}.pruned_init"
                
                task_maps[vla_arena_suite][level][task] = Task(
                    name=task,
                    language=language,
                    problem="vla_arena",
                    problem_folder=problem_folder,
                    bddl_file=bddl_filename,
                    init_states_file=init_states_filename,
                    level=level,
                    level_id=level_id
                )


class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0):
        self.task_embs = None
        self.task_order_index = task_order_index
        self.level_task_maps = {}  # Map from level -> [tasks]

    def _make_benchmark(self):
        # Build level-based task organization
        self.level_task_maps = {0: [], 1: [], 2: []}
        
        # Use the hierarchical structure from vla_arena_task_map
        for level in [0, 1, 2]:
            if level in vla_arena_task_map[self.name]:
                level_tasks = vla_arena_task_map[self.name][level]
                for task_name in level_tasks:
                    if task_name in task_maps[self.name][level]:
                        self.level_task_maps[level].append(task_maps[self.name][level][task_name])
        # Flatten all tasks for backward compatibility
        self.tasks = list(task_maps[self.name].values())
        
        self.n_tasks = 15

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_by_level_id(self, level: int, level_id: int) -> Optional[Task]:
        """
        Get task by level and level_id.
        
        Args:
            level: The difficulty level (0, 1, or 2)
            level_id: The index within that level (0-based)
            
        Returns:
            Task object or None if not found
        """
        if level not in [0, 1, 2]:
            raise ValueError(f"Level must be 0, 1, or 2, got {level}")
        
        if level not in self.level_task_maps:
            return None
            
        level_tasks = self.level_task_maps[level]
        if 0 <= level_id < len(level_tasks):
            return level_tasks[level_id]
        else:
            return None

    def _get_task_file_path(self, level: int, level_id: int, file_type: str, file_extension: str) -> Optional[str]:
        """
        Generic method to get file paths by level and level_id.
        
        Args:
            level: The difficulty level (0, 1, or 2)
            level_id: The index within that level (0-based)
            file_type: Type of file ("bddl_files", "init_states", etc.)
            file_extension: File extension (".bddl", ".pruned_init", etc.)
        """
        task = self.get_task_by_level_id(level, level_id)
        if task is None:
            return None
            
        level_dir = f"level_{task.level}"
        
        if file_type == "bddl_files":
            filename = task.bddl_file
        elif file_type == "init_states":
            filename = task.init_states_file
        else:
            return None
            
        file_path = os.path.join(
            get_vla_arena_path(file_type),
            task.problem_folder,
            level_dir,
            filename
        )
        return file_path

    def get_task_bddl_file_path_by_level_id(self, level: int, level_id: int) -> Optional[str]:
        """Get the bddl file path by level and level_id."""
        return self._get_task_file_path(level, level_id, "bddl_files", ".bddl")

    def get_task_init_states_by_level_id(self, level: int, level_id: int):
        """Get init states by level and level_id."""
        init_states_path = self._get_task_file_path(level, level_id, "init_states", ".pruned_init")
        if init_states_path is None:
            return None
        return torch.load(init_states_path, weights_only=False)

    def get_task_demonstration_by_level_id(self, level: int, level_id: int) -> Optional[str]:
        """Get demonstration path by level and level_id."""
        task = self.get_task_by_level_id(level, level_id)
        if task is None:
            return None
            
        # Extract base task name without level suffix for demo file
        base_task_name = re.sub(r'_L[0-2]$', '', task.name)
        level_dir = f"level_{task.level}"
        demo_path = f"{task.problem_folder}/{level_dir}/{base_task_name}_demo.hdf5"
        return demo_path

    def get_num_tasks_by_level(self, level: int) -> int:
        """Get the number of tasks for a specific level."""
        if level not in [0, 1, 2]:
            raise ValueError(f"Level must be 0, 1, or 2, got {level}")
        return len(self.level_task_maps.get(level, []))

    def get_all_tasks_by_level(self, level: int) -> List[Task]:
        """Get all tasks for a specific level."""
        if level not in [0, 1, 2]:
            raise ValueError(f"Level must be 0, 1, or 2, got {level}")
        return self.level_task_maps.get(level, [])

    def get_task_bddl_file_path(self, level, i):
        """Get the bddl file path with level-based directory structure."""
        return self.get_task_bddl_file_path_by_level_id(level, i)

    def get_task_demonstration(self, i):
        """Get demonstration path by task index."""
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        
        task = self.tasks[i]
        return self.get_task_demonstration_by_level_id(task.level, task.level_id)

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]

    def get_task_init_states(self, level, i):
        """Get init states path with level-based directory structure."""
        return self.get_task_init_states_by_level_id(level, i)
    
    def get_tasks_by_level(self, level):
        """Get all tasks with a specific level."""
        assert level in [0, 1, 2], f"Level must be 0, 1, or 2, got {level}"
        return [task for task in self.tasks if task.level == level]
    
    def get_task_distribution_by_level(self):
        """Get the distribution of tasks across levels."""
        distribution = {0: 0, 1: 0, 2: 0}
        for task in self.tasks:
            distribution[task.level] += 1
        return distribution

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs

    def print_level_summary(self):
        """Print a summary of tasks organized by level."""
        print(f"\n{self.name} Task Summary:")
        print("-" * 50)
        for level in [0, 1, 2]:
            level_tasks = self.level_task_maps.get(level, [])
            print(f"Level {level}: {len(level_tasks)} tasks")
            for i, task in enumerate(level_tasks):
                print(f"  [{i}] {task.name}")
        print(f"\nTotal tasks: {self.n_tasks}")


# Factory function to create benchmark classes
def create_benchmark_class(name):
    """Create a benchmark class with the given name."""
    class BenchmarkClass(Benchmark):
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name = name
            self._make_benchmark()
    
    BenchmarkClass.__name__ = name
    return BenchmarkClass

# Register all benchmark classes using factory
# Organized by category for better readability
benchmark_names = [
    # Safety benchmarks
    "safety_dynamic_obstacles",
    "safety_hazard_avoidance",
    "safety_object_state_preservation",
    "safety_risk_aware_grasping",
    "safety_static_obstacles",
    
    # Robustness benchmarks
    "robustness_dynamic_distractors",
    "robustness_static_distractors",
    "robustness_visual_variations",
    
    # Generalization benchmarks
    "generalization_language_variations",
    "generalization_object_preposition_combinations",
    "generalization_task_workflows",
    "generalization_unseen_objects",
    
    # Other benchmarks
    "long_horizon",

    # Libero benchmarks
    "libero_10",
    "libero_90",
    "libero_spatial",
    "libero_object",
    "libero_goal",
]

# Create and register all benchmark classes
for name in benchmark_names:
    benchmark_class = create_benchmark_class(name)
    register_benchmark(benchmark_class)

# Example usage:
if __name__ == "__main__":
    # Test all benchmarks
    # Organized by category for better readability
    all_benchmarks = [
        # Safety benchmarks
        "safety_dynamic_obstacles",
        "safety_hazard_avoidance",
        "safety_object_state_preservation",
        "safety_risk_aware_grasping",
        "safety_static_obstacles",
        
        # Robustness benchmarks
        "robustness_dynamic_distractors",
        "robustness_static_distractors",
        "robustness_visual_variations",
        
        # Generalization benchmarks
        "generalization_language_variations",
        "generalization_object_preposition_combinations",
        "generalization_task_workflows",
        "generalization_unseen_objects",

        # Other benchmarks
        "long_horizon",

        # LIBERO benchmarks
        "libero_10",
        "libero_90",
        "libero_spatial",
        "libero_object",
        "libero_goal",
    ]
    
    print("Testing all VLA Arena benchmarks:")
    print("=" * 60)
    
    for benchmark_name in all_benchmarks:
        # Get benchmark class
        benchmark_class = get_benchmark(benchmark_name)
        
        # Create instance
        benchmark = benchmark_class()
        
        # Print summary
        print(f"\n{benchmark_name.upper()}")
        print("-" * 40)
        
        # Get task distribution
        distribution = benchmark.get_task_distribution_by_level()
        total = sum(distribution.values())
        
        print(f"Total tasks: {total}")
        for level in [0, 1, 2]:
            print(f"  Level {level}: {distribution[level]} tasks")
        
        # Test accessing a task from each level
        for level in [0, 1, 2]:
            if distribution[level] > 0:
                task = benchmark.get_task_by_level_id(level, 0)
                if task:
                    print(f"  Sample Level {level} task: {task.name}")
    
    print("\n" + "=" * 60)
    print("All benchmarks loaded successfully!")