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
Evaluates a pretrained SmolVLA policy on the VLA-Arena benchmark.
"""

import json
import logging
import math
import os
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
import tqdm
import wandb
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import init_logging

from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv
from vla_arena.vla_arena.utils.eval_init_state import select_init_state_index
from vla_arena.vla_arena.utils.utils import apply_instruction_replacement, load_replacements_dict


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

VLA_ARENA_DUMMY_ACTION = [0.0] * 6 + [-1.0]
VLA_ARENA_ENV_RESOLUTION = 256
DATE_TIME = time.strftime('%Y_%m_%d-%H_%M_%S')
DATE = time.strftime('%Y_%m_%d')


@dataclass
class Args:
    """Evaluation arguments for SmolVLA on VLA-Arena."""
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    policy_path: str = ''
    """Path to the pretrained policy on the Hugging Face Hub or local directory."""
    device: str = 'cuda'
    """Device to use for evaluation."""

    #################################################################################################################
    # VLA-Arena environment-specific parameters
    #################################################################################################################
    # draccus cannot decode generic Iterable; use list for multi-suite configs
    task_suite_name: str | list[str] = 'safety_dynamic_obstacles'
    """Task suite."""
    task_level: int = 0
    """Task level."""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 10
    """Number of rollouts per task."""
    initial_states_path: str = 'DEFAULT'
    """'DEFAULT', or path to initial states JSON file."""
    add_noise: bool = False
    adjust_light: bool = False
    randomize_color: bool = False
    camera_offset: bool = False
    safety: bool = False
    init_state_selection_mode: str = 'first'
    """Init-state selection mode: 'first' or 'episode_idx'."""
    init_state_offset: int = 0
    """Deterministic offset added to selected init-state index."""
    init_state_offset_random: bool = False
    """Whether to add random offset in [0, num_initial_states)."""

    #################################################################################################################
    # Utils
    #################################################################################################################
    use_local_log: bool = True
    """Whether to log to local log file."""
    run_id_note: str | None = None
    """Extra note to add to end of run ID for logging."""
    local_log_dir: str = './experiments/logs'
    """Local directory for eval logs."""
    use_wandb: bool = False
    """Whether to also log results in Weights & Biases."""
    wandb_entity: str = 'your-wandb-entity'
    """Name of WandB entity."""
    wandb_project: str = 'your-wandb-project'
    """Name of WandB project."""

    seed: int = 7
    """Random Seed (for reproducibility)."""

    # Video saving options
    video_out_path: str = f'rollouts/smolvla/{DATE}'
    """Path to save videos."""
    save_video_mode: str = 'first_success_failure'
    """Video saving mode: 'all', 'first_success_failure', 'none'."""

    result_json_path: str | None = None

    #################################################################################################################
    # Instruction replacement parameters
    #################################################################################################################
    use_replacements: bool = True
    """Whether to use instruction replacements."""
    replacements_file: str = 'VLA-Arena/language_replacements'
    """Path to replacements JSON file."""
    replacement_probability: float = 1.0
    """Probability of applying replacement (0.0 to 1.0)."""
    replacement_level: int = 1
    """Level of instruction replacements (from 1 to 4)."""


def setup_logging(cfg: Args):
    """Set up logging to file and optionally to wandb."""
    run_id = f'EVAL-{cfg.task_suite_name}-smolvla-{DATE_TIME}'
    if cfg.run_id_note is not None:
        run_id += f'--{cfg.run_id_note}'
    log_file = None
    local_log_filepath = None
    if cfg.use_local_log:
        os.makedirs(cfg.local_log_dir, exist_ok=True)
        local_log_filepath = os.path.join(cfg.local_log_dir, run_id + '.txt')
        log_file = open(local_log_filepath, 'w')
        logger.info(f'Logging to local log file: {local_log_filepath}')

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file is not None:
        log_file.write(message + '\n')
        log_file.flush()


def initialize_model(cfg: Args):
    """Initialize SmolVLA policy."""
    policy = SmolVLAPolicy.from_pretrained(cfg.policy_path)
    policy.to(cfg.device)
    policy.eval()
    return policy


def load_initial_states(
    cfg: Args, task_suite, task_id: int, task_level=0, log_file=None
):
    """Load initial states for the given task."""
    initial_states = task_suite.get_task_init_states(task_level, task_id)

    if cfg.initial_states_path != 'DEFAULT':
        with open(cfg.initial_states_path) as f:
            all_initial_states = json.load(f)
        log_message(
            f'Using initial states from {cfg.initial_states_path}', log_file
        )
        return initial_states, all_initial_states
    else:
        log_message('Using default initial states', log_file)
        return initial_states, None


def save_rollout_video(
    frames, episode_idx, success, task_description, video_out_path, log_file=None
):
    """Save an MP4 replay of an episode."""
    Path(video_out_path).mkdir(parents=True, exist_ok=True)
    task_segment = (
        task_description.lower()
        .replace(' ', '_')
        .replace('/', '_')
        .replace('.', '_')[:50]
    )
    video_path = (
        Path(video_out_path)
        / f'{DATE_TIME}--episode={episode_idx}--success={success}--task={task_segment}.mp4'
    )
    writer = imageio.get_writer(str(video_path), fps=30)
    for image in frames:
        writer.append_data(image)
    writer.close()
    log_message(f'Saved video to {video_path}', log_file)
    return video_path


def run_episode(
    cfg: Args,
    env,
    task_description: str,
    policy,
    replacements_dict: dict,
    suite_name: str,
    max_steps: int,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    env.reset()
    policy.reset()

    log_message(f'Instruction: {task_description}', log_file)

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(VLA_ARENA_DUMMY_ACTION)

    if cfg.use_replacements:
        replaced_task_description = apply_instruction_replacement(
            task_description, replacements_dict, cfg, logger
        )
        log_message(
            f'Replace Instruction: {task_description} -> {replaced_task_description}',
            log_file,
        )
        task_description = replaced_task_description

    t = 0
    frames = []
    success = False
    cost = 0

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
                .to(cfg.device)
                .unsqueeze(0),
                'observation.images.wrist_image': torch.from_numpy(
                    wrist_img / 255.0
                )
                .permute(2, 0, 1)
                .to(torch.float32)
                .to(cfg.device)
                .unsqueeze(0),
                'observation.state': torch.from_numpy(state)
                .to(torch.float32)
                .to(cfg.device)
                .unsqueeze(0),
                'task': task_description,
            }

            with torch.inference_mode():
                action_tensor = policy.select_action(observation)
            action = action_tensor.cpu().numpy()[0]

            obs, _, done, info = env.step(action)

            if 'cost' in info:
                cost += info['cost']

            if done or t == max_steps - 1:
                if 'cost' in info:
                    if suite_name == 'safety_hazard_avoidance':
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
            log_message(f'Episode error: {e}', log_file)
            break

    return success, frames, cost


def run_task(
    cfg: Args,
    task_suite,
    task_id: int,
    task_level: int,
    policy,
    replacements_dict: dict,
    suite_name: str,
    max_steps: int,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    task = task_suite.get_task_by_level_id(task_level, task_id)

    initial_states, all_initial_states = load_initial_states(
        cfg, task_suite, task_id, task_level, log_file
    )

    env, task_description = _get_vla_arena_env(
        task,
        VLA_ARENA_ENV_RESOLUTION,
        cfg.seed,
        cfg.add_noise,
        cfg.randomize_color,
        cfg.adjust_light,
        cfg.camera_offset,
    )

    if isinstance(task.language, list):
        task_description = task.language[0]
    else:
        task_description = task.language

    task_episodes, task_successes = 0, 0
    first_success_saved = False
    first_failure_saved = False
    total_costs = 0
    success_costs = 0
    failure_costs = 0
    episodes_with_cost = 0
    successes_with_cost = 0
    failures_with_cost = 0
    rng = np.random.default_rng(cfg.seed)

    log_message(
        'Init state selection | '
        f'mode={cfg.init_state_selection_mode} | '
        f'offset={cfg.init_state_offset} | '
        f'offset_random={cfg.init_state_offset_random}',
        log_file,
    )

    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f'\nTask: {task_description}', log_file)

        if cfg.initial_states_path == 'DEFAULT':
            initial_state_idx = select_init_state_index(
                num_initial_states=len(initial_states),
                episode_idx=episode_idx,
                selection_mode=cfg.init_state_selection_mode,
                offset=cfg.init_state_offset,
                offset_random=cfg.init_state_offset_random,
                rng=rng,
            )
            initial_state = (
                initial_states[initial_state_idx]
                if initial_state_idx is not None
                else None
            )
        else:
            initial_states_task_key = task_description.replace(' ', '_')
            episode_key = f'demo_{episode_idx}'

            if not all_initial_states[initial_states_task_key][episode_key]['success']:
                log_message(
                    f'Skipping task {task_id} episode {episode_idx} due to failed expert demo!',
                    log_file,
                )
                continue

            initial_state = np.array(
                all_initial_states[initial_states_task_key][episode_key]['initial_state']
            )

        log_message(f'Starting episode {task_episodes + 1}...', log_file)

        success, frames, cost = run_episode(
            cfg,
            env,
            task_description,
            policy,
            replacements_dict,
            suite_name,
            max_steps,
            initial_state,
            log_file,
        )
        if cost is not None:
            log_message(f'Episode finished with cost {cost}', log_file)

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

        if should_save_video:
            save_rollout_video(
                frames,
                total_episodes,
                success=success,
                task_description=task_description,
                video_out_path=cfg.video_out_path,
                log_file=log_file,
            )

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

    task_success_rate = (
        float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    )
    total_success_rate = (
        float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    )

    log_message(f'Current task success rate: {task_success_rate}', log_file)
    log_message(f'Current total success rate: {total_success_rate}', log_file)
    log_message(f'Current episodes with cost: {episodes_with_cost}', log_file)
    log_message(f'Current total costs: {total_costs}', log_file)
    log_message(f'Current success costs: {success_costs}', log_file)
    log_message(f'Current failure costs: {failure_costs}', log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                f'success_rate/{task_description}': task_success_rate,
                f'num_episodes/{task_description}': task_episodes,
                f'costs/{task_description}': total_costs,
                f'success_costs/{task_description}': success_costs,
                f'failure_costs/{task_description}': failure_costs,
            }
        )

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
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main(cfg: Args | str | Path):
    """Main function to evaluate a trained SmolVLA policy on VLA-Arena benchmark tasks."""
    if isinstance(cfg, (str, Path)):
        config_path = Path(cfg)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found at: {config_path}')

        print(f'Loading configuration from {config_path}...')

        original_argv = sys.argv.copy()
        try:
            sys.argv = [original_argv[0] if original_argv else 'evaluator.py']
            args = draccus.parse(Args, config_path=str(config_path), args=[])
        finally:
            sys.argv = original_argv

    elif isinstance(cfg, Args):
        args = cfg
    else:
        raise ValueError(
            f'Unsupported config type: {type(cfg)}. Expected Args or path string.'
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = initialize_model(args)

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name == 'all':
        # exclude libero from 'all' evaluation
        suite_names: list[str] = [
            name for name in benchmark_dict.keys() 
            if 'libero' not in name.lower()
        ]
    elif isinstance(args.task_suite_name, str):
        suite_names = [args.task_suite_name]
    elif isinstance(args.task_suite_name, Iterable):
        suite_names = list(args.task_suite_name)
    else:
        raise ValueError(
            f'Unsupported task_suite_name type: {type(args.task_suite_name)}'
        )

    tasks_payload: list[dict[str, object]] = []

    replacements_dict = load_replacements_dict(args, logger)

    for suite_name in suite_names:
        if suite_name not in benchmark_dict:
            raise ValueError(
                f'Unknown task suite: {suite_name}. '
                f'Available options are: {list(benchmark_dict.keys())}'
            )

        args_suite = replace(args, task_suite_name=suite_name)
        log_file, local_log_filepath, run_id = setup_logging(args_suite)

        task_suite = benchmark_dict[suite_name]()
        task_level = args_suite.task_level
        num_tasks = (
            10 if suite_name == 'long_horizon' and task_level == 0 else 5
        )
        max_steps = 600 if suite_name == 'long_horizon' and task_level >= 1 else 300

        log_message(f'Task suite: {suite_name}', log_file)
        if args.use_replacements:
            log_message(
                f'Using instruction replacements with probability {args.replacement_probability}',
                log_file,
            )
            log_message(
                f'Loaded {len(replacements_dict)} replacement entries',
                log_file,
            )

        total_episodes = 0
        total_successes = 0
        total_costs = 0
        success_costs = 0
        failure_costs = 0

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
                args_suite,
                task_suite,
                task_id,
                task_level,
                policy,
                replacements_dict,
                suite_name,
                max_steps,
                total_episodes,
                total_successes,
                log_file,
            )
            total_episodes += task_episodes
            total_successes += task_successes
            total_costs += task_total_costs
            success_costs += task_success_costs
            failure_costs += task_failure_costs

        final_success_rate = (
            float(total_successes) / float(total_episodes)
            if total_episodes > 0
            else 0
        )
        average_costs = (
            total_costs / total_episodes if total_episodes > 0 else 0
        )

        log_message(
            f'[{suite_name}] success rate: {final_success_rate:.4f}', log_file
        )
        log_message(f'[{suite_name}] average cost: {average_costs}', log_file)

        if args_suite.use_wandb:
            wandb.log(
                {
                    f'success_rate/{suite_name}': final_success_rate,
                    f'num_episodes/{suite_name}': total_episodes,
                    f'costs/{suite_name}': average_costs,
                }
            )
            wandb.save(local_log_filepath)

        if log_file:
            log_file.close()

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
        result_path = result_dir / f'smolvla_json_{DATE_TIME}.json'
    else:
        result_path = Path(args.result_json_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'name': 'smolvla',
        'tasks': tasks_payload,
    }
    result_path.write_text(json.dumps(payload, indent=2))
    log_message(f'Saved results to {result_path}')

    if len(suite_names) == 1:
        return (
            tasks_payload[0]['data']['sr'][args.task_level],
            tasks_payload[0]['data']['cc'][args.task_level],
        )
    return tasks_payload


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config yaml file',
    )
    args, unknown = parser.parse_known_args()

    init_logging()
    main(cfg=args.config)
