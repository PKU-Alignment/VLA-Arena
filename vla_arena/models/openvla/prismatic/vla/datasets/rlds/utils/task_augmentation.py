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
task_augmentation.py

Contains basic logic for randomly zeroing out keys in the task specification.
"""


import tensorflow as tf

from vla_arena.models.openvla.prismatic.vla.datasets.rlds.utils.data_utils import (
    to_padding,
)


def delete_task_conditioning(traj: dict, keep_image_prob: float) -> dict:
    """
    Randomly drops out either the goal images or the language instruction. Only does something if both of
    these are present.

    Args:
        traj: A dictionary containing trajectory data. Should have a "task" key.
        keep_image_prob: The probability of keeping the goal images. The probability of keeping the language
            instruction is 1 - keep_image_prob.
    """
    if 'language_instruction' not in traj['task']:
        return traj

    image_keys = {
        key
        for key in traj['task'].keys()
        if key.startswith('image_') or key.startswith('depth_')
    }
    if not image_keys:
        return traj

    traj_len = tf.shape(traj['action'])[0]
    should_keep_images = tf.random.uniform([traj_len]) < keep_image_prob
    should_keep_images |= ~traj['task']['pad_mask_dict'][
        'language_instruction'
    ]

    for key in image_keys | {'language_instruction'}:
        should_keep = (
            should_keep_images if key in image_keys else ~should_keep_images
        )
        # pad out the key
        traj['task'][key] = tf.where(
            should_keep,
            traj['task'][key],
            to_padding(traj['task'][key]),
        )
        # zero out the pad mask dict for the key
        traj['task']['pad_mask_dict'][key] = tf.where(
            should_keep,
            traj['task']['pad_mask_dict'][key],
            tf.zeros_like(traj['task']['pad_mask_dict'][key]),
        )

    # when no goal images are present, the goal timestep becomes the final timestep
    traj['task']['timestep'] = tf.where(
        should_keep_images,
        traj['task']['timestep'],
        traj_len - 1,
    )

    return traj
