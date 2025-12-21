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

"""A script to check if any demonstration dataset does not have the exact number of demonstration trajectories"""

from pathlib import Path

import h5py
import numpy as np

from vla_arena.vla_arena import get_vla_arena_path


error_datasets = []
for demo_file_name in Path(get_vla_arena_path('datasets')).rglob('*hdf5'):

    demo_file = h5py.File(demo_file_name)

    count = 0
    for key in demo_file['data'].keys():
        if 'demo' in key:
            count += 1

    if count == 50:
        traj_lengths = []
        action_min = np.inf
        action_max = -np.inf
        for demo_name in demo_file['data'].keys():
            traj_lengths.append(
                demo_file[f'data/{demo_name}/actions'].shape[0]
            )
        traj_lengths = np.array(traj_lengths)
        print(
            f'[info] dataset {demo_file_name} is in tact, test passed \u2714'
        )
        print(np.mean(traj_lengths), ' +- ', np.std(traj_lengths))
        if demo_file['data'].attrs['tag'] == 'vla_arena-v1':
            print('Version correct')

        print('=========================================')

    else:
        print('[error] !!!')
        error_datasets.append(demo_file_name)

if len(error_datasets) > 0:
    print('[error] The following datasets are corrupted:')
    for dataset in error_datasets:
        print(dataset)
