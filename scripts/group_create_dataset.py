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
import json
import os
from pathlib import Path

import h5py
import numpy as np
import robosuite.macros as macros
import robosuite.utils.transform_utils as T

import vla_arena.vla_arena.utils.utils as vla_arena_utils
from vla_arena.vla_arena import get_vla_arena_path
from vla_arena.vla_arena.envs import *


def process_single_demo_file(
    demo_file_path, env_kwargs_template, args, global_demo_counter
):
    """
    Process a single demo HDF5 file and return processed data.

    Args:
        demo_file_path: Original demo file path
        env_kwargs_template: Environment parameters template
        args: Command line arguments
        global_demo_counter: Global demo counter

    Returns:
        List of processed demo data and updated counter
    """

    print(f'\nProcessing file: {demo_file_path}')

    try:
        f = h5py.File(demo_file_path, 'r')
    except Exception as e:
        print(f'Unable to open file {demo_file_path}: {e}')
        return [], global_demo_counter

    # Extract necessary metadata
    try:
        env_name = f['data'].attrs['env']
        env_info = f['data'].attrs['env_info']
        problem_info = json.loads(f['data'].attrs['problem_info'])
        problem_name = problem_info['problem_name']
        language_instruction = problem_info['language_instruction']
        bddl_file_name = f['data'].attrs['bddl_file_name']
        demos = list(f['data'].keys())
    except KeyError as e:
        print(f'File {demo_file_path} missing necessary metadata: {e}')
        f.close()
        return [], global_demo_counter

    # Update environment parameters
    env_kwargs = json.loads(env_info)
    vla_arena_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=not args.not_use_camera_obs,
        has_offscreen_renderer=not args.not_use_camera_obs,
        ignore_done=True,
        use_camera_obs=not args.not_use_camera_obs,
        camera_depths=args.use_depth,
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        camera_segmentations=None,
    )

    # Create environment
    try:
        env = TASK_MAPPING[problem_name](**env_kwargs)
    except Exception as e:
        print(f'Unable to create environment {problem_name}: {e}')
        f.close()
        return [], global_demo_counter

    processed_demos = []
    cap_index = 5

    # Process each episode
    for ep in demos:
        print(f'  Processing {ep}...')

        try:
            # Read model and states
            model_xml = f[f'data/{ep}'].attrs['model_file']
            states = f[f'data/{ep}/states'][()]
            actions = np.array(f[f'data/{ep}/actions'][()])

            # Reset environment
            reset_success = False
            max_reset_attempts = 5
            for attempt in range(max_reset_attempts):
                try:
                    env.reset()
                    reset_success = True
                    break
                except:
                    if attempt == max_reset_attempts - 1:
                        print(
                            f'    Unable to reset environment, skipping {ep}'
                        )
                    continue

            if not reset_success:
                continue

            model_xml = vla_arena_utils.postprocess_model_xml(model_xml, {})

            # Initialize environment state
            init_idx = 0
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(states[init_idx])
            env.sim.forward()
            model_xml = env.sim.model.get_xml()

            camera_names = env.camera_names

            # Containers for collecting data
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            camera_list = {}
            for camera in camera_names:
                camera_list[camera] = {
                    'images': [],
                    'depths': [],
                }
            valid_index = []

            # Replay actions and collect observations
            for j, action in enumerate(actions):
                obs, reward, done, info = env.step(action)

                # Check state consistency
                if j < len(actions) - 1:
                    state_playback = env.sim.get_state().flatten()
                    err = np.linalg.norm(states[j + 1] - state_playback)
                    # if err > 0.01:
                    #     print(f"    [Warning] Playback deviation {err:.2f} at step {j}")

                # Skip first few frames (sensor stabilization)
                if j < cap_index:
                    continue

                valid_index.append(j)

                # Collect proprioception data
                if not args.no_proprio:
                    if 'robot0_gripper_qpos' in obs:
                        gripper_states.append(obs['robot0_gripper_qpos'])
                    joint_states.append(obs['robot0_joint_pos'])
                    ee_states.append(
                        np.hstack(
                            (
                                obs['robot0_eef_pos'],
                                T.quat2axisangle(obs['robot0_eef_quat']),
                            ),
                        ),
                    )

                robot_states.append(env.get_robot_state_vector(obs))

                # Collect image data
                if not args.not_use_camera_obs:
                    if args.use_depth:
                        for camera in camera_names:
                            camera_list[camera]['depths'].append(
                                obs[camera + '_depth']
                            )
                    for camera in camera_names:
                        camera_list[camera]['images'].append(
                            obs[camera + '_image']
                        )

            # Prepare final data
            states = states[valid_index]
            actions = actions[valid_index]
            dones = np.zeros(len(actions)).astype(np.uint8)
            dones[-1] = 1
            rewards = np.zeros(len(actions)).astype(np.uint8)
            rewards[-1] = 1

            # Store processed data
            demo_data = {
                'demo_id': f'demo_{global_demo_counter}',
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
                'robot_states': (
                    np.stack(robot_states, axis=0) if robot_states else None
                ),
                'model_file': model_xml,
                'init_state': states[init_idx] if len(states) > 0 else None,
                'num_samples': len(camera_list[camera_names[0]]['images']),
                'source_file': demo_file_path,
                'original_ep': ep,
            }

            # Add observation data
            if not args.no_proprio and gripper_states:
                demo_data['gripper_states'] = np.stack(gripper_states, axis=0)
                demo_data['joint_states'] = np.stack(joint_states, axis=0)
                demo_data['ee_states'] = np.stack(ee_states, axis=0)
                demo_data['ee_pos'] = demo_data['ee_states'][:, :3]
                demo_data['ee_ori'] = demo_data['ee_states'][:, 3:]

            if not args.not_use_camera_obs:
                for camera in camera_names:
                    if camera_list[camera]['images']:
                        demo_data[camera + '_rgb'] = np.stack(
                            camera_list[camera]['images'], axis=0
                        )

                if args.use_depth:
                    for camera in camera_names:
                        if camera_list[camera]['depths']:
                            demo_data[camera + '_depth'] = np.stack(
                                camera_list[camera]['depths'],
                                axis=0,
                            )

            processed_demos.append(demo_data)
            global_demo_counter += 1

        except Exception as e:
            print(f'    Error processing {ep}: {e}')
            continue

    # Cleanup
    env.close()
    f.close()

    # Return metadata and processed demos
    metadata = {
        'env_name': env_name,
        'problem_info': problem_info,
        'bddl_file_name': bddl_file_name,
        'env_kwargs': env_kwargs,
        'camera_names': camera_names,
    }

    return processed_demos, global_demo_counter, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing original demo HDF5 files (e.g., demonstration_data/xxx/)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory, default is automatically determined based on BDDL file',
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.hdf5',
        help='Filename pattern to process (default: .hdf5)',
    )
    parser.add_argument('--not-use-camera-obs', action='store_true')
    parser.add_argument('--no-proprio', action='store_true')
    parser.add_argument('--use-depth', action='store_true')
    parser.add_argument(
        '--not-recursive',
        action='store_true',
        help='Do not recursively search subdirectories',
    )

    args = parser.parse_args()

    # Find all HDF5 files to process
    if not args.not_recursive:
        demo_files = list(Path(args.input_dir).rglob(args.pattern))
    else:
        demo_files = list(Path(args.input_dir).glob(args.pattern))

    if not demo_files:
        print(f'No files matching {args.pattern} found in {args.input_dir}')
        return

    print(f'Found {len(demo_files)} files to process')

    # Process all files and collect data, grouped by BDDL file
    demos_by_bddl = {}  # {bddl_file_name: [demos]}
    env_kwargs_template = {}
    metadata_by_bddl = {}  # {bddl_file_name: metadata}

    for demo_file in demo_files:
        demos, _, metadata = process_single_demo_file(
            str(demo_file),
            env_kwargs_template,
            args,
            0,  # Each BDDL file counts independently
        )

        if metadata and demos:
            bddl_file_name = metadata['bddl_file_name']
            if bddl_file_name not in demos_by_bddl:
                demos_by_bddl[bddl_file_name] = []
                metadata_by_bddl[bddl_file_name] = metadata
            demos_by_bddl[bddl_file_name].extend(demos)

    # Create an output file for each BDDL file
    for bddl_file_name, demos in demos_by_bddl.items():
        # Generate output path based on original code's naming logic
        demo_dir = args.input_dir  # Input directory as demo_dir
        bddl_base_name = os.path.basename(bddl_file_name)

        if args.output_dir:
            # If output directory is specified, use it
            output_parent_dir = Path(args.output_dir)
            hdf5_file_name = bddl_base_name.replace('.bddl', '_demo.hdf5')
            hdf5_path = output_parent_dir / hdf5_file_name
        else:
            # Otherwise follow original code logic: based on demonstration_data directory structure
            if 'demonstration_data/' in demo_dir:
                relative_dir = demo_dir.split('demonstration_data/')[-1]
            else:
                # If demonstration_data is not in path, use current directory name
                relative_dir = os.path.basename(demo_dir)

            hdf5_file_name = bddl_base_name.replace('.bddl', '_demo.hdf5')
            hdf5_path = os.path.join(
                get_vla_arena_path('datasets'), relative_dir, hdf5_file_name
            )
            hdf5_path = Path(hdf5_path)
            if hdf5_path.exists():
                stem = hdf5_path.stem
                suffix = hdf5_path.suffix
                new_file_name = f'{stem}_1{suffix}'
                hdf5_path = hdf5_path.parent / new_file_name

        output_parent_dir = hdf5_path.parent
        output_parent_dir.mkdir(parents=True, exist_ok=True)

        print(f'\nCreating output file for {bddl_base_name}: {hdf5_path}')

        # Write HDF5 file (using original code structure)
        metadata = metadata_by_bddl[bddl_file_name]

        with h5py.File(str(hdf5_path), 'w') as h5py_f:
            grp = h5py_f.create_group('data')

            # Write attributes (consistent with original code)
            grp.attrs['env_name'] = metadata['env_name']
            grp.attrs['problem_info'] = json.dumps(metadata['problem_info'])
            grp.attrs['macros_image_convention'] = macros.IMAGE_CONVENTION

            # Environment parameters
            problem_name = metadata['problem_info']['problem_name']
            env_args = {
                'type': 1,
                'env_name': metadata['env_name'],
                'problem_name': problem_name,
                'bddl_file': bddl_file_name,
                'env_kwargs': metadata['env_kwargs'],
            }
            grp.attrs['env_args'] = json.dumps(env_args)
            grp.attrs['camera_names'] = metadata['camera_names']

            grp.attrs['bddl_file_name'] = bddl_file_name
            if os.path.exists(bddl_file_name):
                grp.attrs['bddl_file_content'] = open(bddl_file_name).read()

            # Write each demo's data, renumbering
            total_len = 0
            for i, demo_data in enumerate(demos):
                demo_id = f'demo_{i}'  # Renumber starting from 0
                ep_data_grp = grp.create_group(demo_id)

                # Write observation data group
                obs_grp = ep_data_grp.create_group('obs')

                # Proprioception data
                for key in [
                    'gripper_states',
                    'joint_states',
                    'ee_states',
                    'ee_pos',
                    'ee_ori',
                ]:
                    if key in demo_data:
                        obs_grp.create_dataset(key, data=demo_data[key])

                # Image data
                for camera in metadata['camera_names']:
                    for key in [
                        camera + suffix for suffix in ['_rgb', '_depth']
                    ]:
                        if key in demo_data:
                            obs_grp.create_dataset(key, data=demo_data[key])

                # Write action and state data
                ep_data_grp.create_dataset(
                    'actions', data=demo_data['actions']
                )
                ep_data_grp.create_dataset('states', data=demo_data['states'])
                ep_data_grp.create_dataset(
                    'rewards', data=demo_data['rewards']
                )
                ep_data_grp.create_dataset('dones', data=demo_data['dones'])

                if demo_data['robot_states'] is not None:
                    ep_data_grp.create_dataset(
                        'robot_states', data=demo_data['robot_states']
                    )

                # Write attributes
                ep_data_grp.attrs['num_samples'] = demo_data['num_samples']
                ep_data_grp.attrs['model_file'] = demo_data['model_file']
                if demo_data['init_state'] is not None:
                    ep_data_grp.attrs['init_state'] = demo_data['init_state']

                total_len += demo_data['num_samples']

            # Write summary information
            grp.attrs['num_demos'] = len(demos)
            grp.attrs['total'] = total_len

        print(f'Created dataset saved to: {hdf5_path}')
        print(f'Number of demonstrations: {len(demos)}')
        print(f'Total samples: {total_len}')


if __name__ == '__main__':
    main()
