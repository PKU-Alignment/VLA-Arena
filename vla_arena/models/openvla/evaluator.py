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
run_vla_arena_eval.py

Evaluates a trained policy in a VLA-Arena simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import draccus
import numpy as np
import tqdm
import wandb

from vla_arena.models.openvla.experiments.robot.vla_arena.vla_arena_utils import (
    get_vla_arena_dummy_action,
    get_vla_arena_env,
    get_vla_arena_image,
    quat2axisangle,
    save_rollout_video,
)
from vla_arena.vla_arena import benchmark


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
)
from vla_arena.models.openvla.experiments.robot.openvla_utils import (
    get_processor,
)
from vla_arena.models.openvla.experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = 'openvla'                    # Model family
    pretrained_checkpoint: str | Path = ''     # Pretrained checkpoint path

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    unnorm_key: str | Path = 'libero_spatial_no_noops'                # Action un-normalization key
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # VLA-Arena environment-specific parameters
    #################################################################################################################
    task_suite_name: str = 'safety_dynamic_obstacles'  # Task suite
    task_level: int = 1
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                     # Number of rollouts per task
    initial_states_path: str = 'DEFAULT'             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    add_noise: bool = False
    adjust_light: bool = False
    randomize_color: bool = False
    camera_offset: bool = False
    safety: bool = False

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: str | None = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = './experiments/logs'        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = 'your-wandb-entity'          # Name of WandB entity
    wandb_project: str = 'your-wandb-project'        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # Video saving options
    save_video_mode: str = 'first_success_failure'   # Video saving mode: "all", "first_success_failure", "none"

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert (
        cfg.pretrained_checkpoint is not None
    ), 'pretrained_checkpoint must not be None!'

    if 'image_aug' in str(cfg.pretrained_checkpoint):
        assert (
            cfg.center_crop
        ), 'Expecting `center_crop==True` because model was trained with image augmentations!'

    assert not (
        cfg.load_in_8bit and cfg.load_in_4bit
    ), 'Cannot use both 8-bit and 4-bit quantization!'

    # Validate task suite
    # assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == 'openvla':
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    unnorm_key = cfg.unnorm_key

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
    run_id = f'EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}'
    if cfg.run_id_note is not None:
        run_id += f'--{cfg.run_id_note}'

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + '.txt')
    log_file = open(local_log_filepath, 'w')
    logger.info(f'Logging to local log file: {local_log_filepath}')

    # Initialize Weights & Biases logging if enabled
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
    if log_file:
        log_file.write(message + '\n')
        log_file.flush()


def load_initial_states(
    cfg: GenerateConfig, task_suite, task_id: int, task_level=0, log_file=None
):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_level, task_id)

    # If using custom initial states, load them from file
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


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_vla_arena_image(obs, resize_size)

    # Prepare observations dict
    observation = {
        'full_image': img,
        'state': np.concatenate(
            (
                obs['robot0_eef_pos'],
                quat2axisangle(obs['robot0_eef_quat']),
                obs['robot0_gripper_qpos'],
            )
        ),
    }

    return (
        observation,
        img,
    )  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == 'openvla':
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    initial_state=None,
    log_file=None,
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
                obs, reward, done, info = env.step(
                    get_vla_arena_dummy_action(cfg.model_family)
                )
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            action = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
            )

            # Process action
            action = process_action(action, cfg.model_family)

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
    model,
    resize_size,
    processor=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
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
        cfg.model_family,
        resolution=cfg.env_img_res,
        add_noise=cfg.add_noise,
        camera_offset=cfg.camera_offset,
        adjust_light=cfg.adjust_light,
        randomize_color=cfg.randomize_color,
    )
    print(task.language)
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

        # Handle initial state
        if cfg.initial_states_path == 'DEFAULT':
            # Use default initial state
            initial_state = initial_states[0]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(' ', '_')
            episode_key = f'demo_{episode_idx}'

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key][
                'success'
            ]:
                log_message(
                    f'Skipping task {task_id} episode {episode_idx} due to failed expert demo!',
                    log_file,
                )
                continue

            # Get initial state
            initial_state = np.array(
                all_initial_states[initial_states_task_key][episode_key][
                    'initial_state'
                ]
            )

        log_message(f'Starting episode {task_episodes + 1}...', log_file)

        # Run episode
        success, replay_images, cost = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            initial_state,
            log_file,
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
    # Log to wandb if enabled
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


def main(cfg: GenerateConfig | str | Path) -> float:
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
            cfg = draccus.parse(
                GenerateConfig, config_path=str(config_path), args=[]
            )
        finally:
            # Restore original sys.argv
            sys.argv = original_argv

    elif isinstance(cfg, GenerateConfig):
        cfg = cfg
    else:
        raise ValueError(
            f'Unsupported config type: {type(cfg)}. Expected GenerateConfig or path string.'
        )

    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize VLA-Arena task suite
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
            model,
            resize_size,
            processor,
            total_episodes,
            total_successes,
            log_file,
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
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                'success_rate/total': final_success_rate,
                'num_episodes/total': total_episodes,
                'costs/total': average_costs,
                'success_costs/total': average_success_costs,
                'failure_costs/total': average_failure_costs,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return (
        final_success_rate,
        average_costs,
        average_success_costs,
        average_failure_costs,
    )


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

    # Call main with config path string
    main(cfg=args.config)
