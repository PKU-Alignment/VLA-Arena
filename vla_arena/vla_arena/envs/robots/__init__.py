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

import os
from copy import deepcopy

import numpy as np
import robosuite.macros as macros
import robosuite.utils.transform_utils as T
from robosuite.controllers import (
    controller_factory,
    load_part_controller_config,
)
from robosuite.environments.manipulation.manipulation_env import (
    ManipulationEnv,
)
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import gripper_factory
from robosuite.models.tasks import ManipulationTask
from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot
from robosuite.robots.robot import Robot
from robosuite.utils.buffers import DeltaBuffer, RingBuffer
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.transform_utils import mat2quat

from .libero_on_the_ground_panda import LiberoOnTheGroundPanda
from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

TASK_MAPPING = {}


def register_problem(target_class):
    """We design the mapping to be case-INsensitive."""
    TASK_MAPPING[target_class.__name__.lower()] = target_class


ROBOT_CLASS_MAPPING.update(
    {
        'MountedPanda': FixedBaseRobot,
        'OnTheGroundPanda': FixedBaseRobot,
        'LiberoOnTheGroundPanda': FixedBaseRobot,
    },
)
