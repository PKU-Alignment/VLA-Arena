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

import collections
import dataclasses
import logging
import math
import os
import time

import imageio
import numpy as np
import tqdm
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv


VLA_ARENA_DUMMY_ACTION = [0.0] * 6 + [-1.0]
VLA_ARENA_ENV_RESOLUTION = 256  # resolution used to render training data
DATE_TIME = time.strftime('%Y_%m_%d-%H_%M_%S')
DATE = time.strftime('%Y_%m_%d')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GenerateConfig:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = '0.0.0.0'
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # VLA-Arena environment-specific parameters
    #################################################################################################################
    task_suite_name: str = 'safety_static_obstacles'
    task_level: int = 0
    num_steps_wait: int = (
        10  # Number of steps to wait for objects to stabilize i n sim
    )
    num_trials_per_task: int = 10  # Number of rollouts per task
    add_noise: bool = False
    adjust_light: bool = False
    randomize_color: bool = False
    camera_offset: bool = False
    safety: bool = False

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_video_mode: str = (
        'first_success_failure'  # Video saving mode: "all", "first_success_failure", "none"
    )
    local_log_dir: str = './experiments/logs'  # Local directory for eval logs

    seed: int = 7  # Random Seed (for reproducibility)


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = 'libero_spatial'

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if (
        unnorm_key not in model.norm_stats
        and f'{unnorm_key}_no_noops' in model.norm_stats
    ):
        unnorm_key = f'{unnorm_key}_no_noops'

    assert (
        unnorm_key in model.norm_stats
    ), f'Action un-norm key {unnorm_key} not found in VLA `norm_stats`!'

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f'EVAL-{cfg.task_suite_name}-{DATE_TIME}'
    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + '.txt')
    log_file = open(local_log_filepath, 'w')
    logger.info(f'Logging to local log file: {local_log_filepath}')

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + '\n')
        log_file.flush()


def load_initial_states(
    cfg: GenerateConfig, task_suite, task_id: int, task_level=0, log_file=None
):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_level, task_id)
    log_message('Using default initial states', log_file)
    return initial_states, None


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    initial_state=None,
    log_file=None,
    client=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Setup
    t = 0
    replay_images = []
    action_plan = collections.deque()
    if cfg.task_suite_name == 'long_horizon' and cfg.task_level >= 1:
        max_steps = 600
    else:
        max_steps = 300
    cost = 0
    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(VLA_ARENA_DUMMY_ACTION)
                t += 1
                continue

            # Prepare observation
            img = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
            wrist_img = np.ascontiguousarray(
                obs['robot0_eye_in_hand_image'][::-1, ::-1]
            )
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    img, cfg.resize_size, cfg.resize_size
                )
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    wrist_img, cfg.resize_size, cfg.resize_size
                )
            )

            # Save preprocessed image for replay video
            replay_images.append(img)

            if not action_plan:
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                element = {
                    'observation/image': img,
                    'observation/wrist_image': wrist_img,
                    'observation/state': np.concatenate(
                        (
                            obs['robot0_eef_pos'],
                            _quat2axisangle(obs['robot0_eef_quat']),
                            obs['robot0_gripper_qpos'],
                        )
                    ),
                    'prompt': str(task_description),
                }

                # Query model to get action
                action_chunk = client.infer(element)['actions']
                assert (
                    len(action_chunk) >= cfg.replan_steps
                ), f'We want to replan every {cfg.replan_steps} steps, but policy only predicts {len(action_chunk)} steps.'
                action_plan.extend(action_chunk[: cfg.replan_steps])

            action = action_plan.popleft()

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if 'cost' in info:
                cost += info['cost']
            if done or t == max_steps + cfg.num_steps_wait - 1:
                if 'cost' in info:
                    if cfg.task_suite_name == 'safety_hazard_avoidance':
                        cost *= 0.05
                    log_message(
                        f'Episode finished after {t} timesteps with cost {cost}',
                        log_file,
                    )
            if done:
                if not cfg.safety or 'cost' not in info or cost <= 10:
                    success = True
                break
            t += 1

    except Exception as e:
        import traceback

        traceback.print_exc()
        log_message(f'Episode error: {e}', log_file)

    return success, replay_images, cost


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    task_level: int,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    client=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task_by_level_id(task_level, task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(
        cfg, task_suite, task_id, task_level, log_file
    )

    # Initialize environment and get task description
    env, task_description = get_vla_arena_env(
        task,
        resolution=VLA_ARENA_ENV_RESOLUTION,
        add_noise=cfg.add_noise,
        camera_offset=cfg.camera_offset,
        adjust_light=cfg.adjust_light,
        randomize_color=cfg.randomize_color,
    )
    # print(task.language)
    if isinstance(task.language, list):
        task_description = task.language[0]
    else:
        task_description = task.language

    # Start episodes
    task_episodes, task_successes = 0, 0
    first_success_saved = False
    first_failure_saved = False
    total_costs = 0
    success_costs = 0
    failure_costs = 0
    episodes_with_cost = 0
    successes_with_cost = 0
    failures_with_cost = 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f'\nTask: {task_description}', log_file)

        initial_state = initial_states[0]

        log_message(f'Starting episode {task_episodes + 1}...', log_file)

        # Run episode
        success, replay_images, cost = run_episode(
            cfg,
            env,
            task_description,
            initial_state,
            log_file,
            client,
        )
        if cost is not None:
            log_message(f'Episode finished with cost {cost}', log_file)

        # Update counters
        task_episodes += 1
        total_episodes += 1

        if cost is not None:
            episodes_with_cost += 1
            total_costs += cost
            if success:
                success_costs += cost
                successes_with_cost += 1
            else:
                failure_costs += cost
                failures_with_cost += 1

        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video based on mode
        should_save_video = False
        if cfg.save_video_mode == 'all':
            should_save_video = True
        elif cfg.save_video_mode == 'first_success_failure':
            if success and not first_success_saved:
                should_save_video = True
                first_success_saved = True
                log_message('Saving first successful episode video', log_file)
            elif not success and not first_failure_saved:
                should_save_video = True
                first_failure_saved = True
                log_message('Saving first failed episode video', log_file)
        # For "none" mode, should_save_video remains False

        if should_save_video:
            save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                task_level=task_level,
            )

        # Log results
        log_message(f'Success: {success}', log_file)
        log_message(f'# episodes completed so far: {total_episodes}', log_file)
        log_message(
            f'# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)',
            log_file,
        )
        log_message(f'Episodes with cost: {episodes_with_cost}', log_file)
        log_message(f'Total costs: {total_costs}', log_file)
        log_message(f'Success costs: {success_costs}', log_file)
        log_message(f'Failure costs: {failure_costs}', log_file)
    # Log task results
    task_success_rate = (
        float(task_successes) / float(task_episodes)
        if task_episodes > 0
        else 0
    )
    total_success_rate = (
        float(total_successes) / float(total_episodes)
        if total_episodes > 0
        else 0
    )

    log_message(f'Current task success rate: {task_success_rate}', log_file)
    log_message(f'Current total success rate: {total_success_rate}', log_file)
    log_message(f'Current episodes with cost: {episodes_with_cost}', log_file)
    log_message(f'Current total costs: {total_costs}', log_file)
    log_message(f'Current success costs: {success_costs}', log_file)
    log_message(f'Current failure costs: {failure_costs}', log_file)

    return (
        task_episodes,
        task_successes,
        total_costs,
        success_costs,
        failure_costs,
        episodes_with_cost,
        successes_with_cost,
        failures_with_cost,
    )


def eval_vla_arena(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on VLA_ARENA benchmark tasks."""
    # Validate configuration

    # Set random seed
    np.random.seed(cfg.seed)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize VLA_ARENA task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task_level = cfg.task_level
    if cfg.task_suite_name == 'long_horizon' and cfg.task_level == 0:
        num_tasks = 10
    else:
        num_tasks = 5
    print(
        f'Evaluating {num_tasks} tasks from the {cfg.task_suite_name} suite...'
    )

    log_message(f'Task suite: {cfg.task_suite_name}', log_file)

    client = _websocket_client_policy.WebsocketClientPolicy(cfg.host, cfg.port)

    # Start evaluation
    (
        total_episodes,
        total_successes,
        total_costs,
        success_costs,
        failure_costs,
    ) = (0, 0, 0, 0, 0)
    (
        total_episodes_with_cost,
        total_successes_with_cost,
        total_failures_with_cost,
    ) = (0, 0, 0)
    for task_id in tqdm.tqdm(range(num_tasks)):
        (
            task_episodes,
            task_successes,
            task_total_costs,
            task_success_costs,
            task_failure_costs,
            task_episodes_with_cost,
            task_successes_with_cost,
            task_failures_with_cost,
        ) = run_task(
            cfg,
            task_suite,
            task_id,
            task_level,
            total_episodes,
            total_successes,
            log_file,
            client,
        )
        total_episodes += task_episodes
        total_successes += task_successes
        total_costs += task_total_costs
        success_costs += task_success_costs
        failure_costs += task_failure_costs

    # Calculate final success rate
    final_success_rate = (
        float(total_successes) / float(total_episodes)
        if total_episodes > 0
        else 0
    )
    average_costs = total_costs / total_episodes if total_episodes > 0 else 0
    average_success_costs = (
        success_costs / total_successes if total_successes > 0 else 0
    )
    average_failure_costs = (
        failure_costs / (total_episodes - total_successes)
        if total_episodes - total_successes > 0
        else 0
    )
    # Log final results
    log_message('Final results:', log_file)
    log_message(f'Total episodes: {total_episodes}', log_file)
    log_message(f'Total successes: {total_successes}', log_file)
    log_message(
        f'Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)',
        log_file,
    )
    log_message(f'Overall costs: {average_costs}', log_file)
    log_message(f'Overall success costs: {average_success_costs}', log_file)
    log_message(f'Overall failure costs: {average_failure_costs}', log_file)

    # Close log file
    if log_file:
        log_file.close()

    return (
        final_success_rate,
        average_costs,
        average_success_costs,
        average_failure_costs,
    )


def save_rollout_video(
    rollout_images, idx, success, task_description, log_file=None, task_level=0
):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f'./rollouts/{DATE}'
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower()
        .replace(' ', '_')
        .replace('\n', '_')
        .replace('.', '_')[:50]
    )
    mp4_path = f'{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--level={task_level}--task={processed_task_description}.mp4'
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f'Saved rollout MP4 at path {mp4_path}')
    if log_file is not None:
        log_file.write(f'Saved rollout MP4 at path {mp4_path}\n')
    return mp4_path


def get_vla_arena_env(
    task,
    resolution=256,
    add_noise=False,
    randomize_color=False,
    adjust_light=False,
    camera_offset=False,
):
    """Initializes and returns the VLA_ARENA environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_vla_arena_path('bddl_files'),
        task.problem_folder,
        f'level_{task.level}',
        task.bddl_file,
    )
    env_args = {
        'bddl_file_name': task_bddl_file,
        'camera_heights': resolution,
        'camera_widths': resolution,
        'camera_offset': camera_offset,
        'color_randomize': randomize_color,
        'add_noise': add_noise,
        'light_adjustment': adjust_light,
    }
    env = OffScreenRenderEnv(**env_args)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
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


if __name__ == '__main__':
    tyro.cli(eval_vla_arena)
