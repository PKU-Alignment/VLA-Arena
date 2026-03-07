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

"""
This script demonstrates how to evaluate a pretrained smolVLA policy on the VLA-Arena benchmark.
"""

import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import draccus
import imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import init_logging
from tqdm import tqdm

from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv


VLA_ARENA_DUMMY_ACTION = [0.0] * 6 + [-1.0]
VLA_ARENA_ENV_RESOLUTION = 256  # resolution used to render training data
TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
DATE = time.strftime('%Y_%m_%d')


@dataclass
class Args:
    """
    Evaluation arguments for SmolVLA on VLA_Arena.
    """

    # --- Hugging Face arguments ---
    policy_path: str = ''
    """Path to the pretrained policy on the Hugging Face Hub or local directory."""

    # --- VLA-Arena environment-specific parameters ---
    # draccus cannot decode generic Iterable; use list for multi-suite configs
    task_suite_name: str | list[str] = 'safety_dynamic_obstacles'
    """Task suite."""
    task_level: int = 0
    """Task level."""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 10
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = f'rollout/{DATE}'
    """Path to save videos."""
    device: str = 'cuda'
    """Device to use for evaluation."""

    seed: int = 7
    """Random Seed (for reproducibility)"""

    save_video_mode: str = 'first_success_failure'
    add_noise: bool = False
    randomize_color: bool = False
    adjust_light: bool = False
    camera_offset: bool = False

    result_json_path: str | None = None

    # --- Instruction Replacement arguments ---
    use_replacements: bool = True                     # Whether to use instruction replacements
    replacements_file: str = "VLA-Arena/language_replacements"  # Path to replacements JSON file
    replacement_probability: float = 1.0              # Probability of applying replacement (0.0 to 1.0)
    replacement_level: int = 1                        # Level of instruction replacements (from 1 to 4)

def load_replacements_dict(args: Args) -> dict:
    """Load the replacements dictionary from JSON file."""
    if not args.use_replacements:
        return {}
    try:
        if args.replacements_file == 'VLA-Arena/language_replacements':
            filename = f"comprehensive_word_replacements_{args.replacement_level}.json"

            file_path = hf_hub_download(
                repo_id=args.replacements_file, 
                filename=filename,
                repo_type="dataset"
            )

            with open(file_path, 'r') as f:
                replacements_list = json.load(f)
        else:
            with open(args.replacements_file, 'r') as f:
                replacements_list = json.load(f)

        replacements_dict = {}
        for item in replacements_list:
            original_key = item.get('original')  # original instructions
            modified_value = item.get('modified')  # replaced instructions
            if original_key and modified_value:
                if original_key not in replacements_dict:
                    replacements_dict[original_key] = []
                replacements_dict[original_key].append(modified_value)
        
        logging.info(f"Loaded {len(replacements_dict)} replacement entries from {args.replacements_file}")
        return replacements_dict
        
    except FileNotFoundError:
        logging.info(f"Replacements file not found: {args.replacements_file}. Disabling replacements.")
        return {}
    except json.JSONDecodeError as e:
        logging.info(f"Error parsing replacements file: {e}. Disabling replacements.")
        return {}
    except Exception as e:
        logging.info(f"Unexpected error loading replacements: {e}. Disabling replacements.")
        return {}

def apply_instruction_replacement(original_instruction: str, replacements_dict: dict, args: Args) -> str:
    """
    Apply random instruction replacement based on the replacements dictionary.
    
    Args:
        original_instruction: The original instruction string
        replacements_dict: Dictionary mapping normalized instructions to replacement lists
        args: Configuration object containing replacement settings
    
    Returns:
        The potentially replaced instruction string
    """
    if not args.use_replacements or not replacements_dict:
        return original_instruction
    
    # Check if we should apply replacement based on probability
    if random.random() > args.replacement_probability:
        return original_instruction
    
    # Convert instruction to key format: spaces to underscores, lowercase
    instruction_key = original_instruction.lower().replace(" ", "_")
    
    # Check if we have replacements for this instruction
    if instruction_key in replacements_dict:
        replacement_options = replacements_dict[instruction_key]
        if replacement_options:
            # Randomly select one replacement
            selected_replacement = random.choice(replacement_options)
            logging.info(f"Replaced instruction: '{original_instruction}' -> '{selected_replacement}'")
            return selected_replacement.replace("_", " ")
    
    # If no replacement found, return original instruction
    return original_instruction

def eval_vla_arena(args: Args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name == 'all':
        suite_names: list[str] = list(benchmark_dict.keys())
    elif isinstance(args.task_suite_name, str):
        suite_names = [args.task_suite_name]
    elif isinstance(args.task_suite_name, Iterable):
        suite_names = list(args.task_suite_name)
    else:
        raise ValueError(
            f'Unsupported task_suite_name type: {type(args.task_suite_name)}'
        )

    tasks_payload: list[dict[str, object]] = []

    if args.use_replacements:
        replacements_dict = load_replacements_dict(args)
        logging.info(f"Using instruction replacements with probability {args.replacement_probability}")
        logging.info(f"Loaded {len(replacements_dict)} replacement entries")

    for suite_name in suite_names:
        if suite_name not in benchmark_dict:
            raise ValueError(
                f'Unknown task suite: {suite_name}. '
                f'Available options are: {list(benchmark_dict.keys())}'
            )

        args_suite = replace(args, task_suite_name=suite_name)
        task_suite = benchmark_dict[suite_name]()
        task_level = args_suite.task_level
        num_tasks_in_suite = 10 if suite_name == 'long_horizon' and task_level == 0 else 5
        max_steps = 600 if suite_name == 'long_horizon' else 300
        logging.info(f'Task suite: {suite_name}')

        video_out_path = f'{args_suite.video_out_path}/{suite_name}'
        Path(video_out_path).mkdir(parents=True, exist_ok=True)

        total_episodes, total_successes, total_costs = 0, 0, 0

        for task_id in tqdm(range(num_tasks_in_suite), desc='Tasks'):
            task = task_suite.get_task_by_level_id(task_level, task_id)
            initial_states = task_suite.get_task_init_states(
                task_level, task_id
            )

            env, task_description = _get_vla_arena_env(
                task,
                VLA_ARENA_ENV_RESOLUTION,
                args_suite.seed,
                args_suite.add_noise,
                args_suite.randomize_color,
                args_suite.adjust_light,
                args_suite.camera_offset,
            )

            task_episodes, task_successes, task_costs = 0, 0, 0
            first_success_saved, first_failure_saved = False, False
            for episode_idx in tqdm(
                range(args_suite.num_trials_per_task),
                desc=f'Task {task_id}: {task.language}',
                leave=False,
            ):
                if args.use_replacements:
                    replaced_task_description = apply_instruction_replacement(
                        task_description, replacements_dict, args
                    )
                    logging.info(f"Replace Instruction: {task_description} -> {replaced_task_description}")
                    task_description = replaced_task_description

                logging.info(f'\nTask: {task_description}')

                env.reset()
                policy.reset()

                random_offset = rng.integers(0, len(initial_states))
                obs = env.set_init_state(
                    initial_states[
                        (episode_idx + random_offset) % len(initial_states)
                    ]
                )

                for _ in range(args_suite.num_steps_wait):
                    obs, _, _, _ = env.step(VLA_ARENA_DUMMY_ACTION)

                t = 0
                frames = []
                done = False
                cost = 0

                logging.info(f'Starting episode {task_episodes+1}...')
                while t < max_steps:
                    try:
                        wrist_img = np.ascontiguousarray(
                            obs['robot0_eye_in_hand_image'][::-1, ::-1]
                        )
                        agentview_image = np.ascontiguousarray(
                            obs['agentview_image'][::-1, ::-1]
                        )
                        frames.append(agentview_image)

                        state = np.concatenate(
                            (
                                obs['robot0_eef_pos'],
                                _quat2axisangle(obs['robot0_eef_quat']),
                                obs['robot0_gripper_qpos'],
                            )
                        )
                        observation = {
                            'observation.images.image': torch.from_numpy(
                                agentview_image / 255.0
                            )
                            .permute(2, 0, 1)
                            .to(torch.float32)
                            .to(args_suite.device)
                            .unsqueeze(0),
                            'observation.images.wrist_image': torch.from_numpy(
                                wrist_img / 255.0
                            )
                            .permute(2, 0, 1)
                            .to(torch.float32)
                            .to(args_suite.device)
                            .unsqueeze(0),
                            'observation.state': torch.from_numpy(state)
                            .to(torch.float32)
                            .to(args_suite.device)
                            .unsqueeze(0),
                            'task': task_description,
                        }

                        with torch.inference_mode():
                            action_tensor = policy.select_action(observation)
                        action = action_tensor.cpu().numpy()[0]

                        obs, _, done, info = env.step(action)

                        if 'cost' in info:
                            cost += info['cost']
                        if done:
                            if 'cost' in info and suite_name == 'safety_hazard_avoidance':
                                cost *= 0.05
                            logging.info(f'Task success with cost {cost}')
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        logging.error(f'Caught exception: {e}')
                        break

                task_episodes += 1
                total_episodes += 1
                task_costs += cost

                should_save_video = False
                if args_suite.save_video_mode == 'all':
                    should_save_video = True
                elif args_suite.save_video_mode == 'first_success_failure':
                    if done and not first_success_saved:
                        should_save_video = True
                        first_success_saved = True
                        logging.info('Saving first successful episode video')
                    elif not done and not first_failure_saved:
                        should_save_video = True
                        first_failure_saved = True
                        logging.info('Saving first failed episode video')

                video_path = None
                if should_save_video:
                    suffix = 'success' if done else 'failure'
                    task_segment = task_description.replace(' ', '_').replace(
                        '/', '_'
                    )
                    video_path = (
                        Path(video_out_path)
                        / f'{TIME}_rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}.mp4'
                    )
                    fps = 30
                    writer = imageio.get_writer(video_path, fps=fps)

                    for image in frames:
                        writer.append_data(image)
                    writer.close()
                    logging.info(f'Saved video to {video_path}')

                logging.info(f'Success: {done}')
                if total_episodes > 0:
                    logging.info(
                        f'# episodes completed so far: {total_episodes}'
                    )
                    logging.info(
                        f'# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)'
                    )

            total_costs += task_costs
            if task_episodes > 0:
                logging.info(
                    f'Task {task_id} success rate: {float(task_successes) / float(task_episodes):.2f}'
                )
            if total_episodes > 0:
                logging.info(
                    f'Cumulative success rate: {float(total_successes) / float(total_episodes):.2f}'
                )

        logging.info('--- Evaluation finished ---')
        final_success_rate = (
            float(total_successes) / float(total_episodes)
            if total_episodes > 0
            else 0
        )
        average_costs = (
            float(total_costs) / float(total_episodes)
            if total_episodes > 0
            else 0
        )
        logging.info(f'Total success rate: {final_success_rate:.2f}')
        logging.info(f'Average costs: {average_costs:.2f}')
        logging.info(f'Total episodes: {total_episodes}')
        logging.info(f'Total successes: {total_successes}')

        category, has_cc = _suite_category(suite_name)
        sr = [0.0, 0.0, 0.0]
        cc = [0.0, 0.0, 0.0]
        sr[task_level] = final_success_rate
        cc[task_level] = average_costs if has_cc else 0.0
        tasks_payload.append(
            {
                'name': suite_name,
                'category': category,
                'hasCC': has_cc,
                'data': {
                    'sr': sr,
                    'cc': cc,
                },
                'numEpisodes': total_episodes,
                'numSuccesses': total_successes,
            }
        )

    if args.result_json_path is None or str(args.result_json_path).lower() == 'default':
        result_dir = Path('./results')
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"smolvla_json_{TIME}.json"
    else:
        result_path = Path(args.result_json_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'name': 'smolvla',
        'tasks': tasks_payload,
    }
    result_path.write_text(json.dumps(payload, indent=2))
    logging.info(f'Saved results to {result_path}')

    if len(suite_names) == 1:
        return (
            tasks_payload[0]['data']['sr'][args.task_level],
            tasks_payload[0]['data']['cc'][args.task_level],
        )
    return tasks_payload


def _get_vla_arena_env(
    task,
    resolution,
    seed,
    add_noise=False,
    randomize_color=False,
    adjust_light=False,
    camera_offset=False,
):
    """Initializes and returns the VLA-Arena environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        Path(get_vla_arena_path('bddl_files'))
        / task.problem_folder
        / f'level_{task.level}'
        / task.bddl_file
    )
    env_args = {
        'bddl_file_name': str(task_bddl_file),
        'camera_heights': resolution,
        'camera_widths': resolution,
        'camera_offset': camera_offset,
        'color_randomize': randomize_color,
        'add_noise': add_noise,
        'light_adjustment': adjust_light,
    }
    env = OffScreenRenderEnv(**env_args)
    # env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _suite_category(suite_name: str) -> tuple[str, bool]:
    if suite_name.startswith('safety_'):
        return 'Safety', True
    if suite_name.startswith('distractor_'):
        return 'Distractor', False
    if suite_name.startswith('extrapolation_'):
        return 'Extrapolation', False
    if suite_name == 'long_horizon':
        return 'Long Horizon', False
    return 'Other', False


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main(cfg: Args | str | Path):
    """Main function to evaluate a trained policy on VLA-Arena benchmark tasks."""
    # [Config Parsing] Handle cases where config is a path
    if isinstance(cfg, (str, Path)):
        config_path = Path(cfg)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found at: {config_path}')

        print(f'Loading configuration from {config_path}...')

        # Temporarily save sys.argv to avoid draccus parsing command line arguments
        original_argv = sys.argv.copy()
        try:
            # Keep only script name, remove other arguments to avoid draccus parsing command line arguments (e.g., 'eval' subcommand)
            sys.argv = [original_argv[0] if original_argv else 'evaluator.py']
            # Fix: Use config_path, explicitly specify args=[] to avoid parsing from command line
            args = draccus.parse(Args, config_path=str(config_path), args=[])
        finally:
            # Restore original sys.argv
            sys.argv = original_argv

    elif isinstance(cfg, Args):
        args = cfg
    else:
        raise ValueError(
            f'Unsupported config type: {type(cfg)}. Expected Args or path string.'
        )
    eval_vla_arena(args=args)


if __name__ == '__main__':
    import argparse

    # Use argparse to parse --config parameter passed by Launcher
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config yaml file',
    )
    # This allows compatibility with other possible parameters (though currently only config is needed)
    args, unknown = parser.parse_known_args()

    init_logging()
    main(cfg=args.config)
