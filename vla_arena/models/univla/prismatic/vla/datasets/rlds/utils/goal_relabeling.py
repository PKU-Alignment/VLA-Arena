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
goal_relabeling.py

Contains simple goal relabeling logic for BC use-cases where rewards and next_observations are not required.
Each function should add entries to the "task" dict.
"""


import tensorflow as tf

from vla_arena.models.univla.prismatic.vla.datasets.rlds.utils.data_utils import (
    tree_merge,
)


def uniform(traj: dict) -> dict:
    """Relabels with a true uniform distribution over future states."""
    traj_len = tf.shape(tf.nest.flatten(traj['observation'])[0])[0]

    # Select a random future index for each transition i in the range [i + 1, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len) + 1, tf.float32)
    high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # Sometimes there are floating-point errors that cause an out-of-bounds
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # Adds keys to "task" mirroring "observation" keys (`tree_merge` to combine "pad_mask_dict" properly)
    goal = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs), traj['observation']
    )
    traj['task'] = tree_merge(traj['task'], goal)

    return traj
