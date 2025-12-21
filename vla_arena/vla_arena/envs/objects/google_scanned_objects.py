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


class GoogleScannedObject(MujocoXMLObject):
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
                f'assets/stable_scanned_objects/{obj_name}/{obj_name}.xml',
            ),
            name=name,
            joints=joints,
            obj_type='all',
            duplicate_collision_geoms=duplicate_collision_geoms,
        )
        self.category_name = '_'.join(
            re.sub(r'([A-Z])', r' \1', self.__class__.__name__).split(),
        ).lower()
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'
        self.object_properties = {'vis_site_names': {}}


@register_object
class SimpleRack(GoogleScannedObject):
    def __init__(
        self,
        name='simple_rack',
        obj_name='simple_rack',
        joints=[dict(type='free', damping='0.0005')],
    ):
        super().__init__(name, obj_name, joints=joints)
        self.rotation = (0, 0)
        self.rotation_axis = 'x'


@register_object
class WhiteBowl(GoogleScannedObject):
    def __init__(self, name='white_bowl', obj_name='white_bowl'):
        super().__init__(name, obj_name)


@register_object
class RedBowl(GoogleScannedObject):
    def __init__(self, name='red_bowl', obj_name='red_bowl'):
        super().__init__(name, obj_name)


@register_object
class BilliardBalls(GoogleScannedObject):
    def __init__(self, name='billiard_balls', obj_name='billiard_balls'):
        super().__init__(name, obj_name)
        self.rotation = (0, 0)
        self.rotation_axis = 'x'


@register_object
class Mickey(GoogleScannedObject):
    def __init__(self, name='mickey', obj_name='mickey'):
        super().__init__(name, obj_name)
        self.rotation_axis = 'z'
        self.rotation = {
            'x': (np.pi / 2, np.pi / 2),
            'y': (0, 0),
            'z': (np.pi / 2, np.pi / 2),
        }


@register_object
class ToyCar(GoogleScannedObject):
    def __init__(self, name='toy_car', obj_name='toy_car'):
        super().__init__(name, obj_name)
        self.rotation_axis = 'z'


@register_object
class ToyMotorbike(GoogleScannedObject):
    def __init__(self, name='toy_motorbike', obj_name='toy_motorbike'):
        super().__init__(name, obj_name)
        self.rotation_axis = 'z'
        self.rotation = {
            'x': (np.pi / 2, np.pi / 2),
            'y': (0, 0),
            'z': (np.pi / 2, np.pi / 2),
        }


@register_object
class RotatedToyMotorbike(GoogleScannedObject):
    def __init__(self, name='rotated_toy_motorbike', obj_name='toy_motorbike'):
        super().__init__(name, obj_name)
        self.rotation_axis = 'z'
        self.rotation = {'x': (0, 0), 'y': (0, 0), 'z': (np.pi / 2, np.pi / 2)}


@register_object
class RotatedToyMotorbikeX(GoogleScannedObject):
    def __init__(
        self, name='rotated_toy_motorbike_x', obj_name='toy_motorbike'
    ):
        super().__init__(name, obj_name)
        self.rotation_axis = 'z'
        self.rotation = {
            'x': (np.pi / 6, np.pi / 6),
            'y': (0, 0),
            'z': (np.pi / 2, np.pi / 2),
        }


@register_object
class ToyTrain(GoogleScannedObject):
    def __init__(self, name='toy_train', obj_name='toy_train'):
        super().__init__(name, obj_name)
        self.rotation_axis = 'z'
        self.rotation = {
            'x': (np.pi / 4, np.pi / 4),
            'y': (0, 0),
            'z': (np.pi / 2, np.pi / 2),
        }


@register_object
class AkitaBlackBowl(GoogleScannedObject):
    def __init__(self, name='akita_black_bowl', obj_name='akita_black_bowl'):
        super().__init__(name, obj_name)


@register_object
class Pan(GoogleScannedObject):
    def __init__(self, name='pan', obj_name='pan'):
        super().__init__(name, obj_name)
        self.rotation = (0, 0)


@register_object
class Lemon(GoogleScannedObject):
    def __init__(self, name='lemon', obj_name='lemon'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)
        self.rotation = (0, 0)


@register_object
class Candle(GoogleScannedObject):
    def __init__(self, name='candle', obj_name='candle'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'z'


@register_object
class Onion(GoogleScannedObject):
    def __init__(self, name='onion', obj_name='onion'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class OnionN(GoogleScannedObject):
    def __init__(self, name='onion_n', obj_name='onion_n'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Apple(GoogleScannedObject):
    def __init__(self, name='apple', obj_name='apple'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Broccoli(GoogleScannedObject):
    def __init__(self, name='broccoli', obj_name='broccoli'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)
        self.rotation = (np.pi / 4, np.pi / 4)
        self.rotation_axis = 'z'


@register_object
class Banana(GoogleScannedObject):
    def __init__(self, name='banana', obj_name='banana'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class BottledWater(GoogleScannedObject):
    def __init__(self, name='bottled_water', obj_name='bottled_water'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Garlic(GoogleScannedObject):
    def __init__(self, name='garlic', obj_name='garlic'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Lime(GoogleScannedObject):
    def __init__(self, name='lime', obj_name='lime'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Orange(GoogleScannedObject):
    def __init__(self, name='orange', obj_name='orange'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Potato(GoogleScannedObject):
    def __init__(self, name='potato', obj_name='potato'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Tomato(GoogleScannedObject):
    def __init__(self, name='tomato', obj_name='tomato'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class TomatoN(GoogleScannedObject):
    def __init__(self, name='tomato_n', obj_name='tomato_n'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Kiwi(GoogleScannedObject):
    def __init__(self, name='kiwi', obj_name='kiwi'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class KiwiN(GoogleScannedObject):
    def __init__(self, name='kiwi_n', obj_name='kiwi_n'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class BlackBowl(GoogleScannedObject):
    def __init__(
        self,
        name='black_bowl',
        obj_name='black_bowl',
        duplicate_collision_geoms=True,
    ):
        super().__init__(name, obj_name)


@register_object
class Vase(GoogleScannedObject):
    def __init__(self, name='vase', obj_name='vase'):
        super().__init__(name, obj_name)


@register_object
class PorcelainPlate(GoogleScannedObject):
    def __init__(self, name='porcelain_plate', obj_name='porcelain_plate'):
        super().__init__(name, obj_name)


@register_object
class CeramicPlate(GoogleScannedObject):
    def __init__(self, name='ceramic_plate', obj_name='ceramic_plate'):
        super().__init__(name, obj_name)


# @register_object
# class Mirror(GoogleScannedObject):
#     def __init__(self, name="mirror", obj_name="mirror"):
#         super().__init__(name, obj_name)


@register_object
class Plate(GoogleScannedObject):
    def __init__(self, name='plate', obj_name='plate'):
        super().__init__(name, obj_name)


@register_object
class Giftbox(GoogleScannedObject):
    def __init__(self, name='giftbox', obj_name='giftbox'):
        super().__init__(name, obj_name)


@register_object
class Mango(GoogleScannedObject):
    def __init__(self, name='mango', obj_name='mango'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Carrot(GoogleScannedObject):
    def __init__(self, name='carrot', obj_name='carrot'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Cucumber(GoogleScannedObject):
    def __init__(self, name='cucumber', obj_name='cucumber'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class BellPepper(GoogleScannedObject):
    def __init__(self, name='bell_pepper', obj_name='bell_pepper'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'z'


@register_object
class Strawberry(GoogleScannedObject):
    def __init__(self, name='strawberry', obj_name='strawberry'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Peach(GoogleScannedObject):
    def __init__(self, name='peach', obj_name='peach'):
        super().__init__(name, obj_name, duplicate_collision_geoms=True)


@register_object
class Basket(GoogleScannedObject):
    def __init__(self, name='basket', obj_name='basket'):
        super().__init__(name, obj_name)


@register_object
class Counter(GoogleScannedObject):
    def __init__(self, name='counter', obj_name='counter'):
        super().__init__(name, obj_name)
        self.rotation = (0, 0)
        self.rotation_axis = 'x'


@register_object
class Chefmate8Frypan(GoogleScannedObject):
    def __init__(self, name='chefmate_8_frypan', obj_name='chefmate_8_frypan'):
        super().__init__(name, obj_name)


@register_object
class CoffeeMachine(GoogleScannedObject):
    def __init__(self, name='coffee_machine', obj_name='coffee_machine'):
        super().__init__(name, obj_name)


@register_object
class PorcelainBowl(GoogleScannedObject):
    def __init__(self, name='porcelain_bowl', obj_name='porcelain_bowl'):
        super().__init__(name, obj_name)


@register_object
class NewBowl(GoogleScannedObject):
    def __init__(self, name='new_bowl', obj_name='new_bowl'):
        super().__init__(name, obj_name)


@register_object
class NewPlate(GoogleScannedObject):
    def __init__(self, name='new_plate', obj_name='new_plate'):
        super().__init__(name, obj_name)


@register_object
class PinkBowl(GoogleScannedObject):
    def __init__(self, name='pink_bowl', obj_name='pink_bowl'):
        super().__init__(name, obj_name)


@register_object
class WoodenBowl(GoogleScannedObject):
    def __init__(self, name='wooden_bowl', obj_name='wooden_bowl'):
        super().__init__(name, obj_name)


@register_object
class GlazedRimPorcelainRamekin(GoogleScannedObject):
    def __init__(
        self,
        name='glazed_rim_porcelain_ramekin',
        obj_name='glazed_rim_porcelain_ramekin',
    ):
        super().__init__(name, obj_name)


@register_object
class Teapot(GoogleScannedObject):
    def __init__(self, name='teapot', obj_name='teapot'):
        super().__init__(name, obj_name)
        self.rotation = (-np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class DualGas(GoogleScannedObject):
    def __init__(self, name='dual_gas', obj_name='dual_gas'):
        super().__init__(name, obj_name)
        self.rotation = (0, 0)
        self.rotation_axis = 'x'


@register_object
class BasicSleekInduc(GoogleScannedObject):
    def __init__(self, name='basic_sleek_induc', obj_name='basic_sleek_induc'):
        super().__init__(name, obj_name)
        self.rotation = (0, 0)
        self.rotation_axis = 'x'


@register_object
class Box(GoogleScannedObject):
    def __init__(self, name='box', obj_name='box'):
        super().__init__(name, obj_name)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'z'
