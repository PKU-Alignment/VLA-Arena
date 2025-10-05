"""
Regenerates a dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - In the HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
    180 degrees because we observe that the environments return images that are upside down
    on our platform.
    - MODIFIED: Filter demos with != 2 gripper transitions, then progressively add noops after transitions
      in 4-step increments (4, 8, 6) until success.
    - REMOVED: Dataset balancing (no longer enforces 50 demos per task)

Usage:
    python scripts/regenerate_dataset.py \
        --task_suite <TASK SUITE> \
        --raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --target_dir <PATH TO TARGET DIR> \
        --task_levels <LEVEL1> <LEVEL2> ...

    Example (LIBERO-Spatial with single level):
        python scripts/regenerate_dataset.py \
            --task_suite static_obstacles \
            --raw_data_dir ./vla_arena/vla_arena/datasets/spatial \
            --target_dir ./vla_arena/vla_arena/datasets/spatial_no_noops \
            --task_levels 0

    Example (LIBERO-Spatial with multiple levels):
        python scripts/regenerate_dataset.py \
            --task_suite static_obstacles \
            --raw_data_dir ./vla_arena/vla_arena/datasets/spatial \
            --target_dir ./vla_arena/vla_arena/datasets/spatial_no_noops \
            --task_levels 0 1 2

"""

import argparse
import json
import os
import time
import copy
import random

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from vla_arena.vla_arena import benchmark
from vla_arena.vla_arena import get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv


IMAGE_RESOLUTION = 256
MIN_DEMOS_WARNING_THRESHOLD = 20

def get_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]

def get_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_vla_arena_path("bddl_files"), task.problem_folder, f"level_{task.level}", task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    # env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = float(action[-1])
    prev_gripper_action = float(prev_action[-1])
    
    # Use np.allclose for floating point comparison
    return np.linalg.norm(action[:-1]) < threshold and np.allclose(gripper_action, prev_gripper_action)

def has_gripper_transition(action, prev_action):
    """
    Check if there's a gripper transition between two actions.
    Returns True if the gripper action (last dimension) changes from -1 to 1 or 1 to -1.
    """
    if prev_action is None:
        return False
    
    prev_gripper = float(prev_action[-1])
    curr_gripper = float(action[-1])
    
    # Check for transition: -1 to 1 or 1 to -1
    # Use np.allclose for floating point comparison
    is_prev_closed = np.allclose(prev_gripper, -1.0)
    is_prev_open = np.allclose(prev_gripper, 1.0)
    is_curr_closed = np.allclose(curr_gripper, -1.0)
    is_curr_open = np.allclose(curr_gripper, 1.0)
    
    return (is_prev_closed and is_curr_open) or (is_prev_open and is_curr_closed)

def count_gripper_transitions(actions):
    """Count the number of gripper transitions in an action sequence."""
    transitions = 0
    for i in range(len(actions)):
        prev_action = actions[i-1] if i > 0 else None
        if has_gripper_transition(actions[i], prev_action):
            transitions += 1
    return transitions

def preprocess_actions_with_progressive_noops(orig_actions, env, initial_state, max_noops_to_keep=8):
    """
    Preprocess actions with progressive noop retention strategy:
    1. First try with all noops removed
    2. If unsuccessful, progressively keep more noops after transitions (4, 8, 6 steps)
    3. Return the first successful configuration or None if all fail
    
    Returns: (processed_actions, success, noops_kept_after_transitions)
    """
    # Find all gripper transitions
    transition_indices = []
    for i in range(len(orig_actions)):
        prev_action = orig_actions[i-1] if i > 0 else None
        if has_gripper_transition(orig_actions[i], prev_action):
            transition_indices.append(i)
    
    print(f"  Found {len(transition_indices)} gripper transitions at indices: {transition_indices}")
    
    # Try different noop retention strategies
    for noops_to_keep in [4, 8, 12, 16]:
        print(f"  Trying with {noops_to_keep} noops kept after transitions...")
        
        # Build filtered action list
        filtered_actions = []
        indices_to_keep_noops = set()
        
        # Mark indices where we should keep noops (after transitions)
        if noops_to_keep > 0:
            for trans_idx in transition_indices:
                for j in range(trans_idx + 1, min(trans_idx + 1 + noops_to_keep, len(orig_actions))):
                    indices_to_keep_noops.add(j)
        
        # Filter actions
        for i, action in enumerate(orig_actions):
            prev_action = orig_actions[i-1] if i > 0 else None
            
            # Keep action if it's not a noop, or if it's in the keep-noop window
            if not is_noop(action, prev_action) or i in indices_to_keep_noops:
                filtered_actions.append(action)
        
        print(f"    Filtered from {len(orig_actions)} to {len(filtered_actions)} actions")
        try:
        # Test if this configuration works
            replay_data = replay_actions(env, filtered_actions, initial_state)
        except Exception as e:
            print(f"    Error during replay: {e}")
            continue
        
        if replay_data['success']:
            print(f"    SUCCESS with {noops_to_keep} noops kept after transitions!")
            return filtered_actions, True, noops_to_keep, replay_data
        else:
            print(f"    Failed with {noops_to_keep} noops kept")
    
    print(f"    All configurations failed, demo will be filtered out")
    return None, False, -1, None

def replay_actions(env, actions, initial_state):
    """
    Replay a sequence of actions in the environment and collect observations.
    Returns all the collected data and whether the episode was successful.
    """
    # Reset environment and set initial state
    env.reset()
    env.set_init_state(initial_state)
    for _ in range(10):
        obs, reward, done, info = env.step(get_dummy_action())
    
    camera_names = env.env.camera_names
    # Data collection lists
    states = []
    ee_states = []
    gripper_states = []
    joint_states = []
    robot_states = []
    camera_images={}
    for camera in camera_names:
        camera_images[camera] = []
    
    # Replay actions
    for action_idx, action in enumerate(actions):
        # Record state
        states.append(env.sim.get_state().flatten())
        robot_states.append(
            np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
        )
        
        # Record observations
        if "robot0_gripper_qpos" in obs:
            gripper_states.append(obs["robot0_gripper_qpos"])
        joint_states.append(obs["robot0_joint_pos"])
        ee_states.append(
            np.hstack((
                obs["robot0_eef_pos"],
                T.quat2axisangle(obs["robot0_eef_quat"]),
            ))
        )

        for camera in camera_names:
            camera_images[camera].append(obs[camera+"_image"])
        
        # Execute action
        obs, reward, done, info = env.step(action.tolist())
    
    # Convert done to boolean if it's an array
    if isinstance(done, np.ndarray):
        success = bool(done.any() if done.size > 1 else done)
    else:
        success = bool(done)
    
    result = {
        'states': states,
        'ee_states': ee_states,
        'gripper_states': gripper_states,
        'joint_states': joint_states,
        'robot_states': robot_states,
        'actions': actions,
        'success': success
    }
    for camera in camera_names:
        result[camera + '_images'] = camera_images[camera]

    return result


def process_task_without_balancing(task, task_id, task_level, level_raw_dir, env, task_description):
    """
    Process a single task without balancing - keep all successful demonstrations.
    
    Returns:
        - successful_demos: Dictionary of successful demonstrations
        - task_stats: Statistics about the task processing
    """
    task_stats = {
        'total_demos': 0,
        'demos_filtered_transitions': 0,  # Demos with != 2 gripper transitions
        'demos_filtered_failed': 0,       # Demos that failed after all noop strategies
        'final_success': 0,
        'warning_issued': False,
        'noop_strategy_distribution': {0: 0, 4: 0, 8: 0, 12: 0, 16: 0}  # Track which strategies worked
    }
    
    # Get dataset for task
    orig_data_path = os.path.join(level_raw_dir, f"{task.name}_demo.hdf5")
    if not os.path.exists(orig_data_path):
        orig_data_path = os.path.join(level_raw_dir, f"{task.name}_{task_level}_demo.hdf5")
        if not os.path.exists(orig_data_path):
            print(f"Warning: Cannot find raw data file {orig_data_path}. Skipping task.")
            return None, task_stats
    
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]
    
    successful_demos = {}
    
    # Process all original demonstrations
    print(f"\n--- Processing task: {task.name} (Level {task_level}) ---")
    for i in range(len(orig_data.keys())):
        demo_data = orig_data[f"demo_{i}"]
        orig_actions = demo_data["actions"][()]
        orig_states = demo_data["states"][()]
        
        task_stats['total_demos'] += 1
        
        # Check gripper transitions
        num_transitions = count_gripper_transitions(orig_actions)
        if num_transitions > 2:
            print(f'  Demo_{i}: FILTERED (has {num_transitions} transitions, need exactly 2)')
            task_stats['demos_filtered_transitions'] += 1
            continue
        
        print(f'  Processing demo_{i} (2 gripper transitions)')
        
        # Try progressive noop retention
        filtered_actions, success, noops_kept, replay_data = preprocess_actions_with_progressive_noops(
            orig_actions, env, orig_states[0]
        )
        
        if success:
            successful_demos[f"demo_{i}"] = {
                'data': replay_data,
                'original_actions': orig_actions,
                'processed_actions': filtered_actions,
                'initial_state': orig_states[0],
                'actions_removed': len(orig_actions) - len(filtered_actions),
                'noops_kept_after_transitions': noops_kept
            }
            task_stats['noop_strategy_distribution'][noops_kept] += 1
            print(f'    Demo_{i}: SUCCESS (kept {noops_kept} noops after transitions)')
        else:
            task_stats['demos_filtered_failed'] += 1
            print(f'    Demo_{i}: FAILED (filtered out after trying all strategies)')
    
    task_stats['final_success'] = len(successful_demos)
    
    success_count = len(successful_demos)
    print(f"\nFinal success count for {task.name}: {success_count}")
    print(f"  - Filtered for wrong transition count: {task_stats['demos_filtered_transitions']}")
    print(f"  - Filtered for failure after all strategies: {task_stats['demos_filtered_failed']}")
    print(f"  - Noop strategy distribution: {task_stats['noop_strategy_distribution']}")
    
    # Check if we have too few successful demos and issue warning
    if success_count < MIN_DEMOS_WARNING_THRESHOLD:
        task_stats['warning_issued'] = True
        print(f"\n⚠️  WARNING: Task '{task.name}' has only {success_count} successful demonstrations!")
        print(f"⚠️  This is below the minimum threshold of {MIN_DEMOS_WARNING_THRESHOLD}.")
        print(f"⚠️  Consider collecting more demonstrations for this task.")
    
    # Close the original data file
    orig_data_file.close()
    
    return successful_demos, task_stats


def process_level(task_suite, task_level, args, metainfo_json_dict):
    """Process a single task level and return updated metainfo and statistics."""
    print(f"\n{'='*60}")
    print(f"Processing Level {task_level}")
    print(f"{'='*60}")
    
    num_tasks_in_suite = 5
    
    # Statistics for this level
    level_stats = {
        'num_tasks': 0,
        'num_tasks_with_warnings': 0,
        'total_final_success': 0,
        'task_specific_stats': {},
        'skipped_tasks': []
    }
    
    # Create level-specific subdirectory
    level_dir = os.path.join(args.target_dir, f"level_{task_level}")
    
    # Clean up existing directory to ensure all files are regenerated
    if os.path.exists(level_dir):
        import shutil
        print(f"Cleaning up existing level directory: {level_dir}")
        shutil.rmtree(level_dir)
    os.makedirs(level_dir, exist_ok=True)
    
    # Create level-specific raw data directory path
    level_raw_dir = os.path.join(args.raw_data_dir, f"level_{task_level}")
    if not os.path.exists(level_raw_dir):
        level_raw_dir = args.raw_data_dir
        print(f"Note: Using base raw data directory for level {task_level}")
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc=f"Level {task_level} tasks"):
        # Get task in suite
        task = task_suite.get_task_by_level_id(task_level, task_id)
        env, task_description = get_env(task, resolution=IMAGE_RESOLUTION)
        task_description = env.language_instruction
        camera_names = env.env.camera_names
        try:
        # Process task without balancing
            successful_demos, task_stats = process_task_without_balancing(
            task, task_id, task_level, level_raw_dir, env, task_description
        )
        except Exception as e:
            print(f"Error processing task {task.name}: {e}")
            continue
        
        if successful_demos is None:
            level_stats['skipped_tasks'].append(task.name)
            print(f"⚠️  Skipped task: {task.name} (no raw data file found)")
            continue
        
        # Update level statistics
        level_stats['num_tasks'] += 1
        level_stats['total_final_success'] += task_stats['final_success']
        if task_stats['warning_issued']:
            level_stats['num_tasks_with_warnings'] += 1
        
        level_stats['task_specific_stats'][task.name] = task_stats
        
        # Save task-specific HDF5 file
        task_data_path = os.path.join(level_dir, f"{task.name}_demo.hdf5")
        print(f"\nSaving {len(successful_demos)} demos to: {task_data_path}")
        
        with h5py.File(task_data_path, "w") as new_data_file:
            grp = new_data_file.create_group("data")
            grp.attrs['camera_names'] = camera_names
            
            for idx, (demo_id, demo_info) in enumerate(successful_demos.items()):
                replay_data = demo_info['data']
                
                # Prepare data for saving
                actions = replay_data['actions']
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                language_instruction = task_description.encode('utf8')
                # language instruction 和 dones 形状保持一致
                language_instruction = np.array([language_instruction] * len(actions), dtype='S')
                # Save to HDF5
                ep_data_grp = grp.create_group(f"demo_{idx}")
                
                # Save metadata
                ep_data_grp.attrs['actions_removed'] = demo_info['actions_removed']
                ep_data_grp.attrs['noops_kept_after_transitions'] = demo_info['noops_kept_after_transitions']
                
                # Save observation data
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(replay_data['gripper_states'], axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(replay_data['joint_states'], axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(replay_data['ee_states'], axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(replay_data['ee_states'], axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(replay_data['ee_states'], axis=0)[:, 3:])
                for camera in camera_names:
                    obs_grp.create_dataset(camera + "_rgb", data=np.stack(replay_data[camera + '_images'], axis=0))
                
                # Save action and state data
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(replay_data['states']))
                ep_data_grp.create_dataset("robot_states", data=np.stack(replay_data['robot_states'], axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)
                ep_data_grp.create_dataset("language_instruction", data=language_instruction)
                
                # Update metainfo
                task_key = f"level_{task_level}_{task_description.replace(' ', '_')}"
                episode_key = f"demo_{idx}"
                if task_key not in metainfo_json_dict:
                    metainfo_json_dict[task_key] = {}
                metainfo_json_dict[task_key][episode_key] = {
                    "success": True,  # All saved demos are successful
                    "initial_state": demo_info['initial_state'].tolist(),
                    "level": task_level,
                    "actions_removed": demo_info['actions_removed'],
                    "noops_kept_after_transitions": demo_info['noops_kept_after_transitions']
                }
    
    # Print level statistics
    print(f"\n{'='*50}")
    print(f"Level {task_level} Summary:")
    print(f"{'='*50}")
    print(f"  Tasks processed: {level_stats['num_tasks']}")
    if level_stats['skipped_tasks']:
        print(f"  Tasks skipped (no raw data): {level_stats['skipped_tasks']}")
    print(f"  Tasks with warnings (< {MIN_DEMOS_WARNING_THRESHOLD} demos): {level_stats['num_tasks_with_warnings']}")
    print(f"  Total successful demos: {level_stats['total_final_success']}")
    
    print(f"\n  Task-specific summary:")
    for task_name, stats in level_stats['task_specific_stats'].items():
        status = "✓" if stats['final_success'] >= MIN_DEMOS_WARNING_THRESHOLD else "⚠️"
        print(f"    {status} {task_name}: {stats['final_success']} demos")
        print(f"        Filtered: {stats['demos_filtered_transitions']} (wrong transitions), {stats['demos_filtered_failed']} (all strategies failed)")
        print(f"        Success by strategy: 0 noops={stats['noop_strategy_distribution'][0]}, 4={stats['noop_strategy_distribution'][4]}, 8={stats['noop_strategy_distribution'][8]}, 6={stats['noop_strategy_distribution'][6]}")
    
    # Verify all tasks are processed correctly
    expected_files = []
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task_by_level_id(task_level, task_id)
        expected_file = os.path.join(level_dir, f"{task.name}_demo.hdf5")
        expected_files.append((task.name, expected_file))
    
    print(f"\n  File verification:")
    for task_name, expected_file in expected_files:
        if os.path.exists(expected_file):
            with h5py.File(expected_file, "r") as f:
                num_demos = len(f["data"].keys())
                print(f"    ✓ {task_name}: {num_demos} demos")
        else:
            if task_name in level_stats['skipped_tasks']:
                print(f"    - {task_name}: Skipped (no raw data)")
            else:
                print(f"    ❌ {task_name}: Missing file!")
    
    return metainfo_json_dict, level_stats


def main(args):
    print(f"Regenerating {args.task_suite} dataset for levels: {args.task_levels}")
    print(f"Warning threshold: {MIN_DEMOS_WARNING_THRESHOLD} demos")
    print(f"Filtering strategy: Keep demos with exactly 2 gripper transitions")
    print(f"Noop retention: Progressive (4, 8, 12, 16 steps after transitions)")
    print(f"Dataset balancing: DISABLED (keeping all successful demos)")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create target directory
    if os.path.isdir(args.target_dir):
        user_input = input(f"Target directory already exists at path: {args.target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: ")
        if user_input != 'y':
            exit()
        # Clean up the entire target directory to ensure all files are regenerated
        import shutil
        print(f"Cleaning up existing target directory: {args.target_dir}")
        shutil.rmtree(args.target_dir)
    os.makedirs(args.target_dir, exist_ok=True)

    # Prepare JSON file to record metadata
    metainfo_json_out_path = os.path.join(args.target_dir, f"{args.task_suite}_metainfo.json")
    
    metainfo_json_dict = {}
    print(f"Creating new metainfo file at: {metainfo_json_out_path}")
    
    # Add metadata about processing
    metainfo_json_dict["_metadata"] = {
        "task_suite": args.task_suite,
        "levels_processed": args.task_levels,
        "min_demos_warning_threshold": MIN_DEMOS_WARNING_THRESHOLD,
        "filtering_strategy": "Keep demos with exactly 2 gripper transitions",
        "noop_retention": "Progressive (4, 8, 12, 16 steps after transitions)",
        "dataset_balancing": "DISABLED - keeps all successful demos",
        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save initial metainfo file
    with open(metainfo_json_out_path, "w") as f:
        json.dump(metainfo_json_dict, f, indent=2)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    # Overall statistics
    overall_stats = {
        'total_levels': len(args.task_levels),
        'total_tasks': 0,
        'total_tasks_with_warnings': 0,
        'total_final_success': 0,
        'total_demos_filtered_transitions': 0,
        'total_demos_filtered_failed': 0,
        'overall_noop_strategy_distribution': {0: 0, 4: 0, 8: 0, 12: 0, 16: 0}
    }
    
    # Process each level
    for task_level in args.task_levels:
        try:
            metainfo_json_dict, level_stats = process_level(
                task_suite, task_level, args, metainfo_json_dict
            )
            
            # Update overall statistics
            overall_stats['total_tasks'] += level_stats['num_tasks']
            overall_stats['total_tasks_with_warnings'] += level_stats['num_tasks_with_warnings']
            overall_stats['total_final_success'] += level_stats['total_final_success']
            
            # Aggregate filtering stats
            for task_name, task_stats in level_stats['task_specific_stats'].items():
                overall_stats['total_demos_filtered_transitions'] += task_stats['demos_filtered_transitions']
                overall_stats['total_demos_filtered_failed'] += task_stats['demos_filtered_failed']
                for noop_count, count in task_stats['noop_strategy_distribution'].items():
                    overall_stats['overall_noop_strategy_distribution'][noop_count] += count
            
            # Save metainfo after each level (in case of crashes)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)
                
        except Exception as e:
            import traceback
            print(f"Error processing level {task_level}: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            print("Continuing with next level...")
            continue
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    print(f"Total levels processed: {overall_stats['total_levels']}")
    print(f"Total tasks processed: {overall_stats['total_tasks']}")
    print(f"Tasks with warnings (< {MIN_DEMOS_WARNING_THRESHOLD} initial demos): {overall_stats['total_tasks_with_warnings']}")
    print(f"Total successful demos: {overall_stats['total_final_success']}")
    print(f"Demos filtered (wrong transitions): {overall_stats['total_demos_filtered_transitions']}")
    print(f"Demos filtered (all strategies failed): {overall_stats['total_demos_filtered_failed']}")
    
    print(f"\nNoop retention strategy distribution:")
    for noop_count, count in overall_stats['overall_noop_strategy_distribution'].items():
        percentage = (count / max(overall_stats['total_final_success'], 1)) * 100
        print(f"  {noop_count} noops kept: {count} demos ({percentage:.1f}%)")
    
    if overall_stats['total_tasks_with_warnings'] > 0:
        print(f"\n⚠️  WARNING SUMMARY:")
        print(f"⚠️  {overall_stats['total_tasks_with_warnings']} task(s) had fewer than {MIN_DEMOS_WARNING_THRESHOLD} successful demos.")
        print(f"⚠️  Consider collecting more demonstrations for these tasks to improve data quality.")
    
    print(f"\nDataset regeneration complete!")
    print(f"Saved new dataset at: {args.target_dir}")
    print(f"Saved metainfo file at: {metainfo_json_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", type=str, required=True, help="Name of the task suite (e.g., static_obstacles)")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to the raw HDF5 dataset directory")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to the target directory to save the new dataset")
    parser.add_argument("--task_levels", type=int, nargs='+', required=True, help="List of task levels to process (e.g., 0 1 2)")
    args = parser.parse_args()
    
    main(args) 