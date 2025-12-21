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
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
from pathlib import Path

import tensorflow_datasets as tfds
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main(
    data_dir: str = '', output_dir: str = '', *, push_to_hub: bool = False
):
    # Clean up any existing dataset in the output directory
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id='VLA-Arena',
        root=output_path,
        robot_type='panda',
        fps=10,
        features={
            'observation.images.image': {
                'dtype': 'image',
                'shape': (256, 256, 3),
                'names': ['height', 'width', 'rgb'],
            },
            'observation.images.wrist_image': {
                'dtype': 'image',
                'shape': (256, 256, 3),
                'names': ['height', 'width', 'rgb'],
            },
            'observation.state': {
                'dtype': 'float32',
                'shape': (8,),
                'names': {
                    'motors': [
                        'x',
                        'y',
                        'z',
                        'roll',
                        'pitch',
                        'yaw',
                        'gripper',
                        'gripper',
                    ]
                },
            },
            'action': {
                'dtype': 'float32',
                'shape': (7,),
                'names': {
                    'motors': [
                        'x',
                        'y',
                        'z',
                        'roll',
                        'pitch',
                        'yaw',
                        'gripper',
                    ]
                },
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    raw_dataset = tfds.builder_from_directory(data_dir).as_dataset(split='all')
    for episode in raw_dataset:
        for step in episode['steps'].as_numpy_iterator():
            dataset.add_frame(
                {
                    'observation.images.image': step['observation']['image'],
                    'observation.images.wrist_image': step['observation'][
                        'wrist_image'
                    ],
                    'observation.state': step['observation']['state'],
                    'action': step['action'],
                },
                task=step['language_instruction'].decode(),
            )
        dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=['libero', 'panda', 'rlds'],
            private=False,
            push_images=True,
            license='apache-2.0',
        )


if __name__ == '__main__':
    tyro.cli(main)
