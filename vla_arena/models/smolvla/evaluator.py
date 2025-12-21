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
This script demonstrates how to evaluate a pretrained smolVLA policy on the LIBERO benchmark.
"""

import dataclasses
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import draccus
import imageio
import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import init_logging
from tqdm import tqdm

from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
DATE = time.strftime('%Y_%m_%d')


@dataclasses.dataclass
class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """

    # --- Hugging Face arguments ---
    policy_path: str = ''
    """Path to the pretrained policy on the Hugging Face Hub or local directory."""

    # --- VLA-Arena environment-specific parameters ---
    task_suite_name: str = 'safety_dynamic_obstacles'
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


def eval_vla_arena(args: Args) -> None:
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load Policy ---
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()

    # --- Initialize LIBERO task suite ---
    benchmark_dict = benchmark.get_benchmark_dict()
    try:
        task_suite = benchmark_dict[args.task_suite_name]()
    except KeyError:
        raise ValueError(
            f'Unknown task suite: {args.task_suite_name}. '
            f'Available options are: {list(benchmark_dict.keys())}'
        )
    if args.task_suite_name == 'long_horizon' and args.task_level == 0:
        num_tasks_in_suite = 10
    else:
        num_tasks_in_suite = 5
    if args.task_suite_name == 'long_horizon':
        max_steps = 600
    else:
        max_steps = 300
    task_level = args.task_level
    logging.info(f'Task suite: {args.task_suite_name}')

    video_out_path = f'{args.video_out_path}/{args.task_suite_name}'
    Path(video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == 'long_horizon' and args.task_level >= 1:
        max_steps = 600
    else:
        max_steps = 300

    # --- Evaluation Loop ---
    total_episodes, total_successes, total_costs = 0, 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc='Tasks'):
        # Get task
        task = task_suite.get_task_by_level_id(task_level, task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_level, task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_vla_arena_env(
            task,
            LIBERO_ENV_RESOLUTION,
            args.seed,
            args.add_noise,
            args.randomize_color,
            args.adjust_light,
            args.camera_offset,
        )

        # Start episodes
        task_episodes, task_successes, task_costs = 0, 0, 0
        first_success_saved, first_failure_saved = False, False
        for episode_idx in tqdm(
            range(args.num_trials_per_task),
            desc=f'Task {task_id}: {task.language}',
            leave=False,
        ):
            logging.info(f'\nTask: {task_description}')

            # Reset environment and policy
            env.reset()
            policy.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[0])

            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            # Setup
            t = 0
            frames = []
            done = False
            cost = 0

            # Add initial frame
            agentview_image = np.ascontiguousarray(
                obs['agentview_image'][::-1, ::-1]
            )
            # frames.append(agentview_image)
            # import ipdb; ipdb.set_trace()
            logging.info(f'Starting episode {task_episodes+1}...')
            while t < max_steps:
                try:
                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    wrist_img = np.ascontiguousarray(
                        obs['robot0_eye_in_hand_image'][::-1, ::-1]
                    )
                    agentview_image = np.ascontiguousarray(
                        obs['agentview_image'][::-1, ::-1]
                    )
                    frames.append(agentview_image)

                    # Prepare observations dict
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
                        .to(args.device)
                        .unsqueeze(0),
                        'observation.images.wrist_image': torch.from_numpy(
                            wrist_img / 255.0
                        )
                        .permute(2, 0, 1)
                        .to(torch.float32)
                        .to(args.device)
                        .unsqueeze(0),
                        'observation.state': torch.from_numpy(state)
                        .to(torch.float32)
                        .to(args.device)
                        .unsqueeze(0),
                        'task': task_description,
                    }

                    # Query model to get action
                    with torch.inference_mode():
                        action_tensor = policy.select_action(observation)
                    action = action_tensor.cpu().numpy()[0]

                    # Execute action in environment
                    obs, _, done, info = env.step(action)

                    if 'cost' in info:
                        cost += info['cost']
                    if done:
                        if 'cost' in info:
                            if (
                                args.task_suite_name
                                == 'safety_hazard_avoidance'
                            ):
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
            if args.save_video_mode == 'all':
                should_save_video = True
            elif args.save_video_mode == 'first_success_failure':
                if done and not first_success_saved:
                    should_save_video = True
                    first_success_saved = True
                    logging.info('Saving first successful episode video')
                elif not done and not first_failure_saved:
                    should_save_video = True
                    first_failure_saved = True
                    logging.info('Saving first failed episode video')

            if should_save_video:
                # Save a replay video of the episode
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

            # Log current results
            logging.info(f'Success: {done}')
            if total_episodes > 0:
                logging.info(f'# episodes completed so far: {total_episodes}')
                logging.info(
                    f'# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)'
                )

        total_costs += task_costs
        # Log final results for the task
        if task_episodes > 0:
            logging.info(
                f'Task {task_id} success rate: {float(task_successes) / float(task_episodes):.2f}'
            )
        if total_episodes > 0:
            logging.info(
                f'Cumulative success rate: {float(total_successes) / float(total_episodes):.2f}'
            )

    logging.info('--- Evaluation finished ---')
    if total_episodes > 0:
        logging.info(
            f'Total success rate: {float(total_successes) / float(total_episodes):.2f}'
        )
        logging.info(
            f'Average costs: {float(total_costs) / float(total_episodes):.2f}'
        )
    logging.info(f'Total episodes: {total_episodes}')
    logging.info(f'Total successes: {total_successes}')
    cv2.destroyAllWindows()


def _get_vla_arena_env(
    task,
    resolution,
    seed,
    add_noise=False,
    randomize_color=False,
    adjust_light=False,
    camera_offset=False,
):
    """Initializes and returns the LIBERO environment, along with the task description."""
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
