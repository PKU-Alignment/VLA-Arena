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
import importlib
import os

import numpy as np
import tqdm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow_datasets as tfds
from example_transform.transform import transform_step


parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help='name of the dataset to visualize')
args = parser.parse_args()


TARGET_SPEC = {
    'observation': {
        'image': {'shape': (128, 128, 3), 'dtype': np.uint8, 'range': (0, 255)}
    },
    'action': {
        'shape': (8,),
        'dtype': np.float32,
        'range': [
            (-1, -1, -1, -2 * np.pi, -2 * np.pi, -2 * np.pi, -1, 0),
            (+1, +1, +1, +2 * np.pi, +2 * np.pi, +2 * np.pi, +1, 1),
        ],
    },
    'discount': {'shape': (), 'dtype': np.float32, 'range': (0, 1)},
    'reward': {'shape': (), 'dtype': np.float32, 'range': (0, 1)},
    'is_first': {'shape': (), 'dtype': np.bool_, 'range': None},
    'is_last': {'shape': (), 'dtype': np.bool_, 'range': None},
    'is_terminal': {'shape': (), 'dtype': np.bool_, 'range': None},
    'language_instruction': {'shape': (), 'dtype': str, 'range': None},
    'language_embedding': {
        'shape': (512,),
        'dtype': np.float32,
        'range': None,
    },
}


def check_elements(target, values):
    """Recursively checks that elements in `values` match the TARGET_SPEC."""
    for elem in target:
        if isinstance(values[elem], dict):
            check_elements(target[elem], values[elem])
        else:
            if target[elem]['shape']:
                if tuple(values[elem].shape) != target[elem]['shape']:
                    raise ValueError(
                        f"Shape of {elem} should be {target[elem]['shape']} but is {tuple(values[elem].shape)}",
                    )
            if (
                not isinstance(values[elem], bytes)
                and values[elem].dtype != target[elem]['dtype']
            ):
                raise ValueError(
                    f"Dtype of {elem} should be {target[elem]['dtype']} but is {values[elem].dtype}",
                )
            if target[elem]['range'] is not None:
                if isinstance(target[elem]['range'], list):
                    for vmin, vmax, val in zip(
                        target[elem]['range'][0],
                        target[elem]['range'][1],
                        values[elem],
                    ):
                        if not (val >= vmin and val <= vmax):
                            raise ValueError(
                                f"{elem} is out of range. Should be in {target[elem]['range']} but is {values[elem]}.",
                            )
                else:
                    if not (
                        np.all(values[elem] >= target[elem]['range'][0])
                        and np.all(values[elem] <= target[elem]['range'][1])
                    ):
                        raise ValueError(
                            f"{elem} is out of range. Should be in {target[elem]['range']} but is {values[elem]}.",
                        )


# create TF dataset
dataset_name = args.dataset_name
print(f'Visualizing data from dataset: {dataset_name}')
module = importlib.import_module(dataset_name)
ds = tfds.load(dataset_name, split='train')
ds = ds.shuffle(100)

for episode in tqdm.tqdm(ds.take(50)):
    steps = tfds.as_numpy(episode['steps'])
    for step in steps:
        transformed_step = transform_step(step)
        check_elements(TARGET_SPEC, transformed_step)
print("Test passed! You're ready to submit!")
