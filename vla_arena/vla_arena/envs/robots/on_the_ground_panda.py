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

import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import (
    ManipulatorModel,
)
from robosuite.utils.mjcf_utils import xml_path_completion


class OnTheGroundPanda(ManipulatorModel):
    """
    Panda is a sensitive single-arm robot designed by Franka.
    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ['right']

    def __init__(self, idn=0):
        super().__init__(
            xml_path_completion('robots/panda/robot.xml'), idn=idn
        )

        # Set joint damping
        self.set_joint_attribute(
            attrib='damping',
            values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01)),
        )

    @property
    def default_base(self):
        return 'RethinkMount'

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return {'right': 'PandaGripper'}

    @property
    def default_controller_config(self):
        return {'right': 'default_panda'}

    @property
    def init_qpos(self):
        return np.array(
            [
                0,  # Joint 1: Base rotation, keep at 0
                -1.61037389e-01,  # Joint 2: Shoulder joint, keep original value
                0.00,  # Joint 3: Keep at 0
                -2.8,  # Joint 4: Elbow joint, changed from -2.44 to -2.8 (more bent, lower height)
                0.00,  # Joint 5: Keep at 0
                2.3,  # Joint 6: Wrist joint, changed from 2.23 to 2.3 (lower end-effector height)
                np.pi
                / 4,  # Joint 7: End-effector rotation, keep at 45 degrees
            ],
        )

    @property
    def base_xpos_offset(self):
        return {
            'bins': (-0.5, -0.1, 0),
            'empty': (-0.6, 0, -0.7),
            'table': lambda table_length: (-0.16 - table_length / 2, 0, 0),
            'coffee_table': lambda table_length: (
                -0.16 - table_length / 2,
                0,
                -0.3,
            ),
            'living_room_table': lambda table_length: (
                -0.16 - table_length / 2,
                0,
                0.42,
            ),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return 'single'
