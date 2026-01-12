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

import argparse
import os
import time

import numpy as np
import torch

from vla_arena.vla_arena.envs.env_wrapper import OffScreenRenderEnv


parser = argparse.ArgumentParser()
parser.add_argument(
    '--bddl_file', type=str,     
    default='./vla_arena/vla_arena/bddl_files',
    help='Root path',
)
parser.add_argument('--resolution', type=int, default=256, help='Resolution')
parser.add_argument(
    '--output_path',
    type=str,
    default='./vla_arena/vla_arena/init_files',
    help='Output path',
)
parser.add_argument(
    '--root_path',
    type=str,
    default='./vla_arena/vla_arena/bddl_files',
    help='Root path',
)
# Additional argument: default to generating 50 init states
parser.add_argument(
    '--num_inits',
    type=int,
    default=50,
    help='Number of initial states to generate per file',
)
args = parser.parse_args()


def process_single_file_with_retry(bddl_file, relative_path='', max_retries=4):
    """
    Process a single BDDL file with retry mechanism.

    Args:
        bddl_file: Full path to BDDL file
        relative_path: Path relative to input root directory, used to maintain directory structure
        max_retries: Maximum number of retries
    """
    for attempt in range(
        max_retries + 1
    ):  # +1 because it includes the first attempt
        try:
            print(
                f'Processing file: {bddl_file} (Attempt {attempt + 1}/{max_retries + 1})'
            )
            process_single_file(bddl_file, relative_path)
            return  # Successfully processed, return directly

        except Exception as e:
            error_name = e.__class__.__name__

            # Check if it's a RandomizationError
            if (
                'RandomizationError' in error_name
                or 'randomization' in str(e).lower()
            ):
                if attempt < max_retries:
                    print(f'Encountered RandomizationError: {e}')
                    print(
                        f'Retrying... ({attempt + 1}/{max_retries} retries used)'
                    )
                    time.sleep(0.5)  # Brief wait before retry
                    continue
                print(
                    f'Failed after {max_retries} retries due to RandomizationError'
                )
                print(f'Error details: {e}')
                raise e
            # If not RandomizationError, raise exception directly
            print(f'Encountered non-RandomizationError: {error_name}')
            raise e


def process_single_file(bddl_file, relative_path=''):
    """
    Process a single BDDL file.

    Args:
        bddl_file: Full path to BDDL file
        relative_path: Path relative to input root directory, used to maintain directory structure
    """
    resolution = args.resolution
    num_inits = args.num_inits  # Read the requested number of initial states

    """Initialize and return LIBERO environment"""
    env_args = {
        'bddl_file_name': bddl_file,
        'camera_heights': resolution,
        'camera_widths': resolution,
    }
    env = None

    try:
        env = OffScreenRenderEnv(**env_args)

        init_states = []
        
        # Loop to generate the requested number of init states
        # If a RandomizationError occurs here, it will bubble to the outer retry to regenerate the file
        for i in range(num_inits):
            # 1. Load environment (Reset)
            obs = env.reset()
            # print(f'Reset {i+1}/{num_inits} ok') # Optional: reduce log spam

            # 2. Save current initial state
            flattened_state = env.get_sim_state()
            
            if (
                isinstance(flattened_state, np.ndarray)
                and flattened_state.ndim == 1
            ):
                init_states.append(flattened_state)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_inits} states...")

        # If we somehow failed to collect any states, fail fast so the retry mechanism kicks in
        if len(init_states) == 0:
            raise RuntimeError(
                f"No init states were generated for {bddl_file}. Check environment setup."
            )

        # 3. Build output path, maintain original directory structure
        task_name = os.path.basename(bddl_file)
        task_name = task_name.replace('.bddl', '')

        # If there's a relative path, create corresponding directory structure
        if relative_path:
            output_dir = os.path.join(args.output_path, relative_path)
        else:
            output_dir = args.output_path

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f'{task_name}.pruned_init')

        # print(f"Collected {len(init_states)} states")
        # 4. torch.save the init_states
        torch.save(init_states, output_file)

        print(f'Init file saved to {output_file} with {len(init_states)} states')

    finally:
        # 5. Close the environment
        if env is not None:
            env.close()


def process_directory_recursive(directory, root_dir=None):
    """
    Recursively process all BDDL files in a directory.

    Args:
        directory: Current directory being processed
        root_dir: Root directory, used to calculate relative paths
    """
    if root_dir is None:
        root_dir = directory

    # Traverse all files and subdirectories in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path) and item.endswith('.bddl'):
            # Calculate path relative to root directory
            relative_dir = os.path.relpath(directory, root_dir)
            if relative_dir == '.':
                relative_dir = ''

            # Process BDDL file with retry mechanism
            try:
                process_single_file_with_retry(item_path, relative_dir)
            except Exception as e:
                print(f'Error processing {item_path}: {e}')
                print('Skipping this file and continuing with others...')
                continue

        elif os.path.isdir(item_path):
            # Recursively process subdirectory
            process_directory_recursive(item_path, root_dir)


def main():
    bddl_path = args.bddl_file

    if os.path.isfile(bddl_path):
        # If it's a single file, process directly (with retry)
        process_single_file_with_retry(bddl_path)
    elif os.path.isdir(bddl_path):
        # If it's a directory, recursively traverse all .bddl files
        print(f'Recursively processing all .bddl files in {bddl_path}')
        process_directory_recursive(bddl_path, args.root_path)
    else:
        print(f'Error: {bddl_path} is neither a file nor a directory')


if __name__ == '__main__':
    main()