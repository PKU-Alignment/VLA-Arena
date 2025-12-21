# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import defaultdict
from pathlib import Path


def scan_bddl_files_and_generate_dict(
    base_path='./vla_arena/vla_arena/bddl_files',
):
    """
    Scan BDDL file directory and generate task dictionary.

    Args:
        base_path: Root directory path for BDDL files

    Returns:
        dict: Formatted task dictionary
    """
    task_map = {}

    # Define mapping from task suite to directory
    suite_to_dir_mapping = {
        'safety_dynamic_obstacles': 'safety_dynamic_obstacles',
        'safety_hazard_avoidance': 'safety_hazard_avoidance',
        'safety_state_preservation': 'safety_state_preservation',
        'safety_cautious_grasp': 'safety_cautious_grasp',
        'safety_static_obstacles': 'safety_static_obstacles',
        'distractor_dynamic_distractors': 'distractor_dynamic_distractors',
        'distractor_static_distractors': 'distractor_static_distractors',
        'extrapolation_preposition_combinations': 'extrapolation_preposition_combinations',
        'extrapolation_task_workflows': 'extrapolation_task_workflows',
        'extrapolation_unseen_objects': 'extrapolation_unseen_objects',
        'long_horizon': 'long_horizon',
        'libero_10': 'libero_10',
        'libero_90': 'libero_90',
        'libero_spatial': 'libero_spatial',
        'libero_object': 'libero_object',
        'libero_goal': 'libero_goal',
    }

    # Traverse each task suite
    for suite_name, dir_name in suite_to_dir_mapping.items():
        suite_path = Path(base_path) / dir_name

        if not suite_path.exists():
            print(f'Warning: Directory {suite_path} does not exist')
            continue

        # Initialize suite dictionary
        task_map[suite_name] = {0: [], 1: [], 2: []}

        # Traverse three difficulty levels
        for level in [0, 1, 2]:
            level_dir = suite_path / f'level_{level}'

            if 'libero' not in suite_name and not level_dir.exists():
                print(f'Warning: Level directory {level_dir} does not exist')
                continue

            # Scan all .bddl files in this level directory
            bddl_files = sorted(level_dir.glob('*.bddl'))

            for bddl_file in bddl_files:
                # Get filename (without extension)
                task_name = bddl_file.stem

                # Filter out possible duplicate or variant files (e.g., _1, _2 suffixes)
                # If filename ends with _number (but not _L0/L1/L2), skip

                # Add to corresponding level list
                if task_name not in task_map[suite_name][level]:
                    task_map[suite_name][level].append(task_name)

        # Clean up empty lists
        task_map[suite_name] = {
            level: tasks
            for level, tasks in task_map[suite_name].items()
            if tasks
        }

    return task_map


def generate_python_dict_code(task_map):
    """
    Generate Python dictionary code string.

    Args:
        task_map: Task dictionary

    Returns:
        str: Formatted Python code
    """
    code_lines = ['vla_arena_task_map = {']

    for suite_idx, (suite_name, levels) in enumerate(task_map.items()):
        code_lines.append(f'    "{suite_name}": {{')

        for level_idx, (level, tasks) in enumerate(sorted(levels.items())):
            code_lines.append(f'        {level}: [')

            # Group by scene (if vla_arena_90)
            if suite_name == 'vla_arena_90' and len(tasks) > 10:
                # Group by scene prefix
                scene_groups = defaultdict(list)
                for task in tasks:
                    # Extract scene prefix (e.g., KITCHEN_SCENE1)
                    match = re.match(r'^([A-Z_]+_SCENE\d+)', task)
                    if match:
                        scene_prefix = match.group(1)
                        scene_groups[scene_prefix].append(task)
                    else:
                        scene_groups['OTHER'].append(task)

                # Output by scene
                for scene_idx, (scene, scene_tasks) in enumerate(
                    sorted(scene_groups.items())
                ):
                    if scene_idx > 0:
                        code_lines.append(
                            ''
                        )  # Add empty line to separate different scenes
                    code_lines.append(f'            # {scene} tasks')
                    for task in sorted(scene_tasks):
                        code_lines.append(f'            "{task}",')
            else:
                # Normal output
                for task in tasks:
                    code_lines.append(f'            "{task}",')

            code_lines.append('        ],')

        code_lines.append('    },')

    code_lines.append('}')

    return '\n'.join(code_lines)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Scan BDDL files and generate task dictionary'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default='./vla_arena/vla_arena/bddl_files',
        help='Root directory path for BDDL files',
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='./vla_arena/vla_arena/benchmark/vla_arena_suite_task_map.py',
        help='Output file path',
    )
    parser.add_argument(
        '--print-only',
        action='store_true',
        help='Only print result, do not save file',
    )

    args = parser.parse_args()

    # Scan files and generate dictionary
    print(f'Scanning BDDL files in: {args.base_path}')
    task_map = scan_bddl_files_and_generate_dict(args.base_path)

    # Generate code
    code = generate_python_dict_code(task_map)

    # Add helper functions
    helper_functions = '''

# Helper function to get all tasks for a suite (flattened from all levels)
def get_all_tasks_for_suite(suite_name):
    """Get all tasks for a suite, combining all levels."""
    if suite_name not in vla_arena_task_map:
        return []

    all_tasks = []
    for level in [0, 1, 2]:
        if level in vla_arena_task_map[suite_name]:
            all_tasks.extend(vla_arena_task_map[suite_name][level])
    return all_tasks


# Helper function to get tasks by level for a suite
def get_tasks_by_level(suite_name, level):
    """Get tasks for a specific suite and level."""
    if suite_name not in vla_arena_task_map:
        return []

    if level not in vla_arena_task_map[suite_name]:
        return []

    return vla_arena_task_map[suite_name][level]


# Helper function to count tasks per level for a suite
def count_tasks_per_level(suite_name):
    """Count tasks per level for a specific suite."""
    if suite_name not in vla_arena_task_map:
        return {}

    counts = {}
    for level in [0, 1, 2]:
        if level in vla_arena_task_map[suite_name]:
            counts[level] = len(vla_arena_task_map[suite_name][level])
        else:
            counts[level] = 0
    return counts


# Print summary statistics
if __name__ == "__main__":
    print("VLA Arena Task Map Summary:")
    print("-" * 50)

    for suite_name in vla_arena_task_map:
        counts = count_tasks_per_level(suite_name)
        total = sum(counts.values())
        print(f"\\n{suite_name}:")
        print(f"  Total tasks: {total}")
        print(f"  Level 0: {counts[0]} tasks")
        print(f"  Level 1: {counts[1]} tasks")
        print(f"  Level 2: {counts[2]} tasks")
'''

    full_code = code + helper_functions

    if args.print_only:
        print('\n' + '=' * 60)
        print('Generated task map:')
        print('=' * 60)
        print(full_code)
    else:
        # Save to file
        with open(args.output_file, 'w') as f:
            f.write(full_code)
        print(f'\nTask map saved to: {args.output_file}')

    # Print statistics
    print('\n' + '=' * 60)
    print('Statistics:')
    print('=' * 60)
    for suite_name, levels in task_map.items():
        total = sum(len(tasks) for tasks in levels.values())
        print(f'\n{suite_name}:')
        print(f'  Total tasks: {total}')
        for level in sorted(levels.keys()):
            print(f'  Level {level}: {len(levels[level])} tasks')


if __name__ == '__main__':
    main()
