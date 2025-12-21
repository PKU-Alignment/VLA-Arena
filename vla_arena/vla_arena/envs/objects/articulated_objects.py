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
import pathlib
import re

import numpy as np
from robosuite.models.objects import MujocoXMLObject


absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from vla_arena.vla_arena.envs.base_object import (
    register_object,
    register_visual_change_object,
)


class ArticulatedObject(MujocoXMLObject):
    def __init__(
        self,
        name,
        obj_name,
        joints=[dict(type='free', damping='0.0005')],
        duplicate_collision_geoms=False,
    ):
        super().__init__(
            os.path.join(
                str(absolute_path),
                f'assets/articulated_objects/{obj_name}.xml',
            ),
            name=name,
            joints=joints,
            obj_type='all',
            duplicate_collision_geoms=duplicate_collision_geoms,
        )
        self.category_name = '_'.join(
            re.sub(r'([A-Z])', r' \1', self.__class__.__name__).split(),
        ).lower()
        self.rotation = (np.pi / 4, np.pi / 2)
        self.rotation_axis = 'x'

        articulation_object_properties = {
            'default_open_ranges': [],
            'default_close_ranges': [],
        }
        self.object_properties = {
            'articulation': articulation_object_properties,
            'vis_site_names': {},
        }

    def is_open(self, qpos):
        raise NotImplementedError

    def is_close(self, qpos):
        raise NotImplementedError


@register_object
class Microwave(ArticulatedObject):
    def __init__(
        self,
        name='microwave',
        obj_name='microwave',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)

        self.object_properties['articulation']['default_open_ranges'] = [
            -2.094,
            -1.3,
        ]
        self.object_properties['articulation']['default_close_ranges'] = [
            -0.005,
            0.0,
        ]

    def is_open(self, qpos):
        if qpos < max(
            self.object_properties['articulation']['default_open_ranges']
        ):
            return True
        return False

    def is_close(self, qpos):
        if qpos > min(
            self.object_properties['articulation']['default_close_ranges']
        ):
            return True
        return False


@register_object
class SlideCabinet(ArticulatedObject):
    def __init__(
        self,
        name='slide_cabinet',
        obj_name='slide_cabinet',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class Window(ArticulatedObject):
    def __init__(
        self,
        name='window',
        obj_name='window',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.z_on_table = 0.13


@register_object
class Faucet(ArticulatedObject):
    def __init__(
        self,
        name='faucet',
        obj_name='faucet',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class BasinFaucet(ArticulatedObject):
    def __init__(
        self, name='basin_faucet', obj_name='basin_faucet', joints=None
    ):
        super().__init__(name, obj_name, joints)


@register_object
class ShortCabinet(ArticulatedObject):
    def __init__(
        self,
        name='short_cabinet',
        obj_name='short_cabinet',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)

        self.object_properties['articulation']['default_open_ranges'] = [
            0.10,
            0.16,
        ]
        self.object_properties['articulation']['default_close_ranges'] = [
            -0.005,
            0.0,
        ]

    def is_open(self, qpos):
        if qpos > min(
            self.object_properties['articulation']['default_open_ranges']
        ):
            return True
        return False

    def is_close(self, qpos):
        if qpos < max(
            self.object_properties['articulation']['default_close_ranges']
        ):
            return True
        return False


@register_object
class ShortFridge(ArticulatedObject):
    def __init__(
        self,
        name='short_fridge',
        obj_name='short_fridge',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)

        self.object_properties['articulation']['default_open_ranges'] = [
            2.0,
            2.7,
        ]
        self.object_properties['articulation']['default_close_ranges'] = [
            -0.005,
            0.0,
        ]

    def is_open(self, qpos):
        if qpos > min(
            self.object_properties['articulation']['default_open_ranges']
        ):
            return True
        return False

    def is_close(self, qpos):
        if qpos < max(
            self.object_properties['articulation']['default_close_ranges']
        ):
            return True
        return False

    # Sample initial joint positions for random door open or door closed


@register_object
class WoodenCabinet(ArticulatedObject):
    def __init__(
        self,
        name='wooden_cabinet',
        obj_name='wooden_cabinet',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.object_properties['articulation']['default_open_ranges'] = [
            -0.16,
            -0.14,
        ]
        self.object_properties['articulation']['default_close_ranges'] = [
            0.0,
            0.005,
        ]

    def is_open(self, qpos):
        if qpos < max(
            self.object_properties['articulation']['default_open_ranges']
        ):
            return True
        return False

    def is_close(self, qpos):
        if qpos > min(
            self.object_properties['articulation']['default_close_ranges']
        ):
            return True
        return False


@register_object
class SimpleGas(ArticulatedObject):
    def __init__(
        self,
        name='simple_gas',
        obj_name='simple_gas',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (0, 0)
        self.rotation_axis = 'y'


@register_object
class WhiteCabinet(ArticulatedObject):
    def __init__(
        self,
        name='white_cabinet',
        obj_name='white_cabinet',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.object_properties['articulation']['default_open_ranges'] = [
            -0.16,
            -0.14,
        ]
        self.object_properties['articulation']['default_close_ranges'] = [
            0.0,
            0.005,
        ]

    def is_open(self, qpos):
        if qpos < max(
            self.object_properties['articulation']['default_open_ranges']
        ):
            return True
        return False

    def is_close(self, qpos):
        if qpos > min(
            self.object_properties['articulation']['default_close_ranges']
        ):
            return True
        return False


@register_object
@register_visual_change_object
class FlatStove(ArticulatedObject):
    def __init__(
        self,
        name='flat_stove',
        obj_name='flat_stove',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (0, 0)
        self.rotation_axis = 'y'

        tracking_sites_dict = {}
        tracking_sites_dict['burner'] = (self.naming_prefix + 'burner', False)
        self.object_properties['vis_site_names'].update(tracking_sites_dict)
        self.object_properties['articulation']['default_turnon_ranges'] = [
            0.5,
            2.1,
        ]
        self.object_properties['articulation']['default_turnoff_ranges'] = [
            -0.005,
            0.0,
        ]

    def turn_on(self, qpos):
        if qpos >= min(
            self.object_properties['articulation']['default_turnon_ranges']
        ):
            # TODO: Set visualization sites to be true
            self.object_properties['vis_site_names']['burner'] = (
                self.naming_prefix + 'burner',
                True,
            )
            return True
        self.object_properties['vis_site_names']['burner'] = (
            self.naming_prefix + 'burner',
            False,
        )
        return False

    def turn_off(self, qpos):
        if qpos < max(
            self.object_properties['articulation']['default_turnoff_ranges']
        ):
            self.object_properties['vis_site_names']['burner'] = (
                self.naming_prefix + 'burner',
                False,
            )
            return True
        self.object_properties['vis_site_names']['burner'] = (
            self.naming_prefix + 'burner',
            True,
        )
        return False


@register_object
class Ball(ArticulatedObject):
    def __init__(
        self,
        name='ball',
        obj_name='ball',
        joints=None,
    ):
        super().__init__(name, obj_name, joints)


@register_object
class WaterBall(ArticulatedObject):
    def __init__(
        self,
        name='water_ball',
        obj_name='water_ball',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


# @register_object
# class Vase(ArticulatedObject):
#     def __init__(
#         self,
#         name="vase",
#         obj_name="vase",
#         joints=[dict(type="free", damping="0.0005")],
#     ):
#         super().__init__(name, obj_name, joints, duplicate_collision_geoms=True)
