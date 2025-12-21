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
import robosuite.utils.transform_utils as transform_utils


class BaseObjectState:
    def __init__(self):
        pass

    def get_geom_state(self):
        raise NotImplementedError

    def check_contact(self, other):
        raise NotImplementedError

    def check_contain(self, other):
        raise NotImplementedError

    def get_joint_state(self):
        raise NotImplementedError

    def is_open(self):
        raise NotImplementedError

    def is_close(self):
        raise NotImplementedError

    def get_size(self):
        raise NotImplementedError

    def check_ontop(self, other):
        raise NotImplementedError


class ObjectState(BaseObjectState):
    def __init__(self, env, object_name, is_fixture=False):
        self.env = env
        self.object_name = object_name
        self.is_fixture = is_fixture
        self.query_dict = (
            self.env.fixtures_dict
            if self.is_fixture
            else self.env.objects_dict
        )
        self.object_state_type = 'object'
        self.has_turnon_affordance = hasattr(
            self.env.get_object(self.object_name), 'turn_on'
        )

    def get_geom_state(self):
        object_pos = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        object_quat = self.env.sim.data.body_xquat[
            self.env.obj_body_id[self.object_name]
        ]
        return {'pos': object_pos, 'quat': object_quat}

    def check_contact(self, other):
        object_1 = self.env.get_object(self.object_name)
        object_2 = self.env.get_object(other.object_name)
        return self.env.check_contact(object_1, object_2)

    def check_force(self, other):
        object_1 = self.env.get_object(self.object_name)
        object_2 = self.env.get_object(other.object_name)
        return self.env.check_force(object_1, object_2)

    def check_distance(self, other):
        object_1 = self.env.get_object(self.object_name)
        object_2 = self.env.get_object(other.object_name)
        return self.env.check_distance(object_1, object_2)

    def check_gripper_distance(self):
        object_1 = self.env.get_object(self.object_name)
        return self.env.check_gripper_distance(object_1)

    def check_gripper_distance_part(self, geom_name_1):
        object_1 = self.env.get_object(self.object_name)
        return self.env.check_gripper_distance_part(object_1, geom_name_1)

    def check_contain(self, other):
        object_1 = self.env.get_object(self.object_name)
        object_1_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        object_2 = self.env.get_object(other.object_name)
        object_2_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]
        return object_1.in_box(object_1_position, object_2_position)

    def get_joint_state(self):
        # Return None if joint state does not exist
        joint_states = []
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            joint_states.append(self.env.sim.data.qpos[qpos_addr])
        return joint_states

    def check_ontop(self, other):
        this_object = self.env.get_object(self.object_name)
        this_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        other_object = self.env.get_object(other.object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]
        return (
            (this_object_position[2] <= other_object_position[2])
            and self.check_contact(other)
            and (
                np.linalg.norm(
                    this_object_position[:2] - other_object_position[:2]
                )
                < 0.07
            )
        )

    def set_joint(self, qpos=1.5):
        for joint in self.env.get_object(self.object_name).joints:
            self.env.sim.data.set_joint_qpos(joint, qpos)

    def is_open(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if self.env.get_object(self.object_name).is_open(qpos):
                return True
        return False

    def is_close(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if not (self.env.get_object(self.object_name).is_close(qpos)):
                return False
        return True

    def turn_on(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if self.env.get_object(self.object_name).turn_on(qpos):
                return True
        return False

    def turn_off(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if not (self.env.get_object(self.object_name).turn_off(qpos)):
                return False
        return True

    def update_state(self):
        if self.has_turnon_affordance:
            self.turn_on()

    def fall(self):
        """
        Detect if an object has fallen based on position and orientation changes.

        This method checks if an object has fallen by comparing its current state
        with its original state. A fall is detected if:
        - Position changes significantly (total displacement > 0.1m, height drop > 0.05m, or XY displacement > 0.15m)
        - Orientation changes significantly (rotation about any axis exceeds threshold)

        Returns:
            bool: True if object has fallen, False otherwise
        """
        # Get original and current states
        original_pos = self.env.object_original_pos.get(self.object_name)
        original_quat = self.env.object_original_quat.get(self.object_name)

        if original_pos is None or original_quat is None:
            return False

        current_pos = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        current_quat = self.env.sim.data.body_xquat[
            self.env.obj_body_id[self.object_name]
        ]

        # Check position changes
        pos_diff = np.linalg.norm(current_pos - original_pos)
        height_drop = (
            original_pos[2] - current_pos[2]
        )  # Positive value indicates drop
        xy_diff = np.linalg.norm(current_pos[:2] - original_pos[:2])

        # Position fall detection
        pos_fall = (pos_diff > 0.1) or (height_drop > 0.05) or (xy_diff > 0.15)

        # Check orientation changes
        quat_diff = transform_utils.quat_multiply(
            current_quat,
            transform_utils.quat_inverse(original_quat),
        )
        quat_diff_euler = transform_utils.quat2axisangle(quat_diff)

        # Orientation fall detection (rotation about any axis exceeds threshold)
        rotation_fall = (
            (abs(quat_diff_euler[0]) > 0.2)
            or (abs(quat_diff_euler[1]) > 0.2)
            or (abs(quat_diff_euler[2]) > 0.5)
        )

        return pos_fall or rotation_fall

    def check_gripper_contact(self):
        object_1 = self.env.get_object(self.object_name)
        return self.env.check_gripper_contact(object_1)

    def check_in_contact_part(self, object_name, geom_name_1, geom_name_2):
        object_1 = self.env.get_object(self.object_name)
        object_2 = self.env.get_object(object_name)
        return self.env.check_in_contact_part(
            object_1, object_2, geom_name_1, geom_name_2
        )

    def check_gripper_contact_part(self, geom_name_1):
        object_1 = self.env.get_object(self.object_name)
        return self.env.check_gripper_contact_part(object_1, geom_name_1)


class SiteObjectState(BaseObjectState):
    """
    This is to make site based objects to have the same API as normal Object State.
    """

    def __init__(self, env, object_name, parent_name, is_fixture=False):
        self.env = env
        self.object_name = object_name
        self.parent_name = parent_name
        self.is_fixture = self.parent_name in self.env.fixtures_dict
        self.query_dict = (
            self.env.fixtures_dict
            if self.is_fixture
            else self.env.objects_dict
        )
        self.object_state_type = 'site'

    def get_geom_state(self):
        object_pos = self.env.sim.data.get_site_xpos(self.object_name)
        object_quat = transform_utils.mat2quat(
            self.env.sim.data.get_site_xmat(self.object_name)
        )
        return {'pos': object_pos, 'quat': object_quat}

    def check_contain(self, other):
        this_object = self.env.object_sites_dict[self.object_name]
        this_object_position = self.env.sim.data.get_site_xpos(
            self.object_name
        )
        this_object_mat = self.env.sim.data.get_site_xmat(self.object_name)

        other_object = self.env.get_object(other.object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]
        return this_object.in_box(
            this_object_position, this_object_mat, other_object_position
        )

    def check_contact(self, other):
        """
        There is no dynamics for site objects, so we return true all the time.
        """
        return True

    def check_ontop(self, other):
        this_object = self.env.object_sites_dict[self.object_name]
        if hasattr(this_object, 'under'):
            this_object_position = self.env.sim.data.get_site_xpos(
                self.object_name
            )
            this_object_mat = self.env.sim.data.get_site_xmat(self.object_name)
            other_object = self.env.get_object(other.object_name)
            other_object_position = self.env.sim.data.body_xpos[
                self.env.obj_body_id[other.object_name]
            ]
            # print(self.object_name, this_object_position)
            # print(other_object_position)

            parent_object = self.env.get_object(self.parent_name)
            if parent_object is None:
                return this_object.under(
                    this_object_position,
                    this_object_mat,
                    other_object_position,
                )
            return this_object.under(
                this_object_position,
                this_object_mat,
                other_object_position,
            ) and self.env.check_contact(parent_object, other_object)
        return True

    def set_joint(self, qpos=1.5):
        for joint in self.env.object_sites_dict[self.object_name].joints:
            self.env.sim.data.set_joint_qpos(joint, qpos)

    def is_open(self):
        for joint in self.env.object_sites_dict[self.object_name].joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if self.env.get_object(self.parent_name).is_open(qpos):
                return True
        return False

    def is_close(self):
        for joint in self.env.object_sites_dict[self.object_name].joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if not (self.env.get_object(self.parent_name).is_close(qpos)):
                return False
        return True
