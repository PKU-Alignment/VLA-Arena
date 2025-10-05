import os
import re
from pathlib import Path
from collections import defaultdict

def scan_bddl_files_and_generate_dict(base_path="./vla_arena/vla_arena/bddl_files"):
    """
    扫描BDDL文件目录并生成任务字典
    
    Args:
        base_path: BDDL文件的根目录路径
    
    Returns:
        dict: 格式化的任务字典
    """
    task_map = {}
    
    # 定义任务套件到目录的映射
    suite_to_dir_mapping = {
        "safety_dynamic_obstacles": "safety_dynamic_obstacles",
        "safety_hazard_avoidance": "safety_hazard_avoidance",
        "safety_object_state_preservation": "safety_object_state_preservation",
        "safety_risk_aware_grasping": "safety_risk_aware_grasping",
        "safety_static_obstacles": "safety_static_obstacles",
        "robustness_dynamic_distractors": "robustness_dynamic_distractors",
        "robustness_static_distractors": "robustness_static_distractors",
        "robustness_visual_variations": "robustness_visual_variations",
        "generalization_language_variations": "generalization_language_variations",
        "generalization_object_preposition_combinations": "generalization_object_preposition_combinations",
        "generalization_task_workflows": "generalization_task_workflows",
        "generalization_unseen_objects": "generalization_unseen_objects",
        "long_horizon": "long_horizon",
        "libero_10": "libero_10",
        "libero_90": "libero_90",
        "libero_spatial": "libero_spatial",
        "libero_object": "libero_object",
        "libero_goal": "libero_goal",
    }
    
    # 遍历每个任务套件
    for suite_name, dir_name in suite_to_dir_mapping.items():
        suite_path = Path(base_path) / dir_name
        
        if not suite_path.exists():
            print(f"Warning: Directory {suite_path} does not exist")
            continue
        
        # 初始化套件字典
        task_map[suite_name] = {0: [], 1: [], 2: []}
        
        # 遍历三个难度等级
        for level in [0, 1, 2]:
            level_dir = suite_path / f"level_{level}"
            
            if not level_dir.exists():
                print(f"Warning: Level directory {level_dir} does not exist")
                continue
            
            # 扫描该等级目录下的所有.bddl文件
            bddl_files = sorted(level_dir.glob("*.bddl"))
            
            for bddl_file in bddl_files:
                # 获取文件名（不含扩展名）
                task_name = bddl_file.stem
                
                # 过滤掉可能的重复或变体文件（如 _1, _2 等后缀）
                # 如果文件名以 _数字 结尾（但不是 _L0/L1/L2），则跳过
                
                # 添加到对应等级的列表中
                if task_name not in task_map[suite_name][level]:
                    task_map[suite_name][level].append(task_name)
        
        # 清理空列表
        task_map[suite_name] = {
            level: tasks for level, tasks in task_map[suite_name].items() if tasks
        }
    
    return task_map


def generate_python_dict_code(task_map):
    """
    生成Python字典的代码字符串
    
    Args:
        task_map: 任务字典
    
    Returns:
        str: 格式化的Python代码
    """
    code_lines = ["vla_arena_task_map = {"]
    
    for suite_idx, (suite_name, levels) in enumerate(task_map.items()):
        code_lines.append(f'    "{suite_name}": {{')
        
        for level_idx, (level, tasks) in enumerate(sorted(levels.items())):
            code_lines.append(f'        {level}: [')
            
            # 按场景分组（如果是vla_arena_90）
            if suite_name == "vla_arena_90" and len(tasks) > 10:
                # 按场景前缀分组
                scene_groups = defaultdict(list)
                for task in tasks:
                    # 提取场景前缀（如 KITCHEN_SCENE1）
                    match = re.match(r'^([A-Z_]+_SCENE\d+)', task)
                    if match:
                        scene_prefix = match.group(1)
                        scene_groups[scene_prefix].append(task)
                    else:
                        scene_groups['OTHER'].append(task)
                
                # 按场景输出
                for scene_idx, (scene, scene_tasks) in enumerate(sorted(scene_groups.items())):
                    if scene_idx > 0:
                        code_lines.append("")  # 添加空行分隔不同场景
                    code_lines.append(f'            # {scene} tasks')
                    for task in sorted(scene_tasks):
                        code_lines.append(f'            "{task}",')
            else:
                # 普通输出
                for task in tasks:
                    code_lines.append(f'            "{task}",')
            
            code_lines.append('        ],')
        
        code_lines.append('    },')
    
    code_lines.append('}')
    
    return '\n'.join(code_lines)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='扫描BDDL文件并生成任务字典')
    parser.add_argument(
        '--base-path', 
        type=str, 
        default="./vla_arena/vla_arena/bddl_files",
        help='BDDL文件的根目录路径'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default="./vla_arena/vla_arena/benchmark/vla_arena_suite_task_map.py",
        help='输出文件路径'
    )
    parser.add_argument(
        '--print-only',
        action='store_true',
        help='只打印结果，不保存文件'
    )
    
    args = parser.parse_args()
    
    # 扫描文件并生成字典
    print(f"Scanning BDDL files in: {args.base_path}")
    task_map = scan_bddl_files_and_generate_dict(args.base_path)
    
    # 生成代码
    code = generate_python_dict_code(task_map)
    
    # 添加辅助函数
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
        print("\n" + "="*60)
        print("Generated task map:")
        print("="*60)
        print(full_code)
    else:
        # 保存到文件
        with open(args.output_file, 'w') as f:
            f.write(full_code)
        print(f"\nTask map saved to: {args.output_file}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)
    for suite_name, levels in task_map.items():
        total = sum(len(tasks) for tasks in levels.values())
        print(f"\n{suite_name}:")
        print(f"  Total tasks: {total}")
        for level in sorted(levels.keys()):
            print(f"  Level {level}: {len(levels[level])} tasks")


if __name__ == "__main__":
    main()