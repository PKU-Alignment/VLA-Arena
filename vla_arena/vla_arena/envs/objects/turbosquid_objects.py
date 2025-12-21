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

from vla_arena.vla_arena.envs.base_object import register_object


class TurbosquidObjects(MujocoXMLObject):
    def __init__(
        self, name, obj_name, joints=[dict(type='free', damping='0.0005')]
    ):
        super().__init__(
            os.path.join(
                str(absolute_path),
                f'assets/turbosquid_objects/{obj_name}/{obj_name}.xml',
            ),
            name=name,
            joints=joints,
            obj_type='all',
            duplicate_collision_geoms=False,
        )
        self.category_name = '_'.join(
            re.sub(r'([A-Z])', r' \1', self.__class__.__name__).split(),
        ).lower()
        self.rotation = (0, 0)
        self.rotation_axis = 'x'
        self.object_properties = {'vis_site_names': {}}


@register_object
class WoodenTray(TurbosquidObjects):
    def __init__(
        self,
        name='wooden_tray',
        obj_name='wooden_tray',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class WhiteStorageBox(TurbosquidObjects):
    def __init__(
        self,
        name='white_storage_box',
        obj_name='white_storage_box',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (0, 0)
        self.rotation_axis = 'y'


@register_object
class WoodenShelf(TurbosquidObjects):
    def __init__(
        self,
        name='wooden_shelf',
        obj_name='wooden_shelf',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class WoodenTwoLayerShelf(TurbosquidObjects):
    def __init__(
        self,
        name='wooden_two_layer_shelf',
        obj_name='wooden_two_layer_shelf',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (np.pi / 4, np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class WineRack(TurbosquidObjects):
    def __init__(
        self,
        name='wine_rack',
        obj_name='wine_rack',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class WineBottle(TurbosquidObjects):
    def __init__(
        self,
        name='wine_bottle',
        obj_name='wine_bottle',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class DiningSetGroup(TurbosquidObjects):
    """This dining set group is mostly for visualization"""

    def __init__(
        self,
        name='dining_set_group',
        obj_name='dining_set_group',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class BowlDrainer(TurbosquidObjects):
    def __init__(
        self,
        name='bowl_drainer',
        obj_name='bowl_drainer',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class MokaPot(TurbosquidObjects):
    def __init__(
        self,
        name='moka_pot',
        obj_name='moka_pot',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class BlackBook(TurbosquidObjects):
    def __init__(
        self,
        name='black_book',
        obj_name='black_book',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 4)


@register_object
class YellowBook(TurbosquidObjects):
    def __init__(
        self,
        name='yellow_book',
        obj_name='yellow_book',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 4)


@register_object
class RedCoffeeMug(TurbosquidObjects):
    def __init__(
        self,
        name='red_coffee_mug',
        obj_name='red_coffee_mug',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 2)


@register_object
class GreenMug(TurbosquidObjects):
    def __init__(
        self,
        name='green_mug',
        obj_name='green_mug',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class BlueMug(TurbosquidObjects):
    def __init__(
        self,
        name='blue_mug',
        obj_name='blue_mug',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class PatternedMug(TurbosquidObjects):
    def __init__(
        self,
        name='patterned_mug',
        obj_name='patterned_mug',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class BrownMug(TurbosquidObjects):
    def __init__(
        self,
        name='brown_mug',
        obj_name='brown_mug',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class DeskCaddy(TurbosquidObjects):
    def __init__(
        self,
        name='desk_caddy',
        obj_name='desk_caddy',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)


@register_object
class PorcelainMug(TurbosquidObjects):
    def __init__(
        self,
        name='porcelain_mug',
        obj_name='porcelain_mug',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 2)
        # is that right?
        # I think it is right, because the mug is flat in the initial state, no, the mug is vertical, so why not set the quat to (0, 0, 0, 1)?
        # self.object_properties['quat'] = (0, 0, 0, 1)
        # In the initial state, the mug is rotated 90 degrees around the x axis, so the quat is (0, 0, 0, 1)
        self.object_properties['initial_quat'] = (0, 0, 0, 1)

    def check_flat(self):
        # check if the mug is rotated 90 degrees around the x axis or y axis, why not use the initial quat?
        quat = self.get_geom_state()['quat']
        return quat[0] > 0.9 or quat[1] > 0.9


@register_object
class WhiteYellowMug(TurbosquidObjects):
    def __init__(
        self,
        name='white_yellow_mug',
        obj_name='white_yellow_mug',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 2, -np.pi / 2)


@register_object
class CuttingBoard(TurbosquidObjects):
    def __init__(
        self,
        name='cutting_board_no_rotation',
        obj_name='cutting_board',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (-np.pi / 8, -np.pi / 8)


@register_object
class CuttingBoardNoRotation(TurbosquidObjects):
    def __init__(
        self,
        name='cutting_board_no_rotation',
        obj_name='cutting_board',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints)
