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

import json

import h5py
import numpy as np


def get_dataset_info(dataset_path, filter_key=None, verbose=True):
    # extract demonstration list from file
    all_filter_keys = None
    f = h5py.File(dataset_path, 'r')
    if filter_key is not None:
        # use the demonstrations from the filter key instead
        print(f'NOTE: using filter key {filter_key}')
        demos = sorted(
            [
                elem.decode('utf-8')
                for elem in np.array(f[f'mask/{filter_key}'])
            ]
        )
    else:
        # use all demonstrations
        demos = sorted(list(f['data'].keys()))

        # extract filter key information
        if 'mask' in f:
            all_filter_keys = {}
            for fk in f['mask']:
                fk_demos = sorted(
                    [
                        elem.decode('utf-8')
                        for elem in np.array(f[f'mask/{fk}'])
                    ]
                )
                all_filter_keys[fk] = fk_demos

    # put demonstration list in increasing episode order
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # extract length of each trajectory in the file
    traj_lengths = []
    action_min = np.inf
    action_max = -np.inf
    for ep in demos:
        traj_lengths.append(f[f'data/{ep}/actions'].shape[0])
        action_min = min(action_min, np.min(f[f'data/{ep}/actions'][()]))
        action_max = max(action_max, np.max(f[f'data/{ep}/actions'][()]))
    traj_lengths = np.array(traj_lengths)

    problem_info = json.loads(f['data'].attrs['problem_info'])

    language_instruction = ''.join(problem_info['language_instruction'])
    # report statistics on the data
    print('')
    print(f'total transitions: {np.sum(traj_lengths)}')
    print(f'total trajectories: {traj_lengths.shape[0]}')
    print(f'traj length mean: {np.mean(traj_lengths)}')
    print(f'traj length std: {np.std(traj_lengths)}')
    print(f'traj length min: {np.min(traj_lengths)}')
    print(f'traj length max: {np.max(traj_lengths)}')
    print(f'action min: {action_min}')
    print(f'action max: {action_max}')
    print('language instruction: {}'.format(language_instruction.strip('"')))
    print('')
    print('==== Filter Keys ====')
    if all_filter_keys is not None:
        for fk in all_filter_keys:
            print(f'filter key {fk} with {len(all_filter_keys[fk])} demos')
    else:
        print('no filter keys')
    print('')
    if verbose:
        if all_filter_keys is not None:
            print('==== Filter Key Contents ====')
            for fk in all_filter_keys:
                print(
                    f'filter_key {fk} with {len(all_filter_keys[fk])} demos: {all_filter_keys[fk]}',
                )
        print('')
    env_meta = json.loads(f['data'].attrs['env_args'])
    print('==== Env Meta ====')
    print(json.dumps(env_meta, indent=4))
    print('')

    print('==== Dataset Structure ====')
    for ep in demos:
        print(
            'episode {} with {} transitions'.format(
                ep, f[f'data/{ep}'].attrs['num_samples']
            )
        )
        for k in f[f'data/{ep}']:
            if k in ['obs', 'next_obs']:
                print(f'    key: {k}')
                for obs_k in f[f'data/{ep}/{k}']:
                    shape = f[f'data/{ep}/{k}/{obs_k}'].shape
                    print(
                        f'        observation key {obs_k} with shape {shape}'
                    )
            elif isinstance(f[f'data/{ep}/{k}'], h5py.Dataset):
                key_shape = f[f'data/{ep}/{k}'].shape
                print(f'    key: {k} with shape {key_shape}')

        if not verbose:
            break

    f.close()
