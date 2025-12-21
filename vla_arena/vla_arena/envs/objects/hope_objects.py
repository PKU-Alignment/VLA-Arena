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


class HopeBaseObject(MujocoXMLObject):
    def __init__(self, name, obj_name, duplicate_collision_geoms=False):
        super().__init__(
            os.path.join(
                str(absolute_path),
                f'assets/stable_hope_objects/{obj_name}/{obj_name}.xml',
            ),
            name=name,
            joints=[dict(type='free', damping='0.0005')],
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
class AlphabetSoup(HopeBaseObject):
    def __init__(self, name='alphabet_soup', obj_name='alphabet_soup'):
        super().__init__(name, obj_name)
        self.rotation_axis = 'z'


@register_object
class BbqSauce(HopeBaseObject):
    def __init__(self, name='bbq_sauce', obj_name='bbq_sauce'):
        super().__init__(name, obj_name)


@register_object
class Butter(HopeBaseObject):
    def __init__(self, name='butter', obj_name='butter'):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = 'x'


@register_object
class Cherries(HopeBaseObject):
    def __init__(self, name='cherries', obj_name='cherries'):
        super().__init__(name, obj_name)


@register_object
class ChocolatePudding(HopeBaseObject):
    def __init__(self, name='chocolate_pudding', obj_name='chocolate_pudding'):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = 'x'


@register_object
class Cookies(HopeBaseObject):
    def __init__(self, name='cookies', obj_name='cookies'):
        super().__init__(name, obj_name)


@register_object
class Corn(HopeBaseObject):
    def __init__(self, name='corn', obj_name='corn'):
        super().__init__(name, obj_name)


@register_object
class CreamCheese(HopeBaseObject):
    def __init__(self, name='cream_cheese', obj_name='cream_cheese'):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = 'x'


@register_object
class Ketchup(HopeBaseObject):
    def __init__(self, name='ketchup', obj_name='ketchup'):
        super().__init__(name, obj_name)
        self.rotation = {
            'x': (np.pi / 2, np.pi / 2),
            'z': (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None


@register_object
class MacaroniAndCheese(HopeBaseObject):
    def __init__(
        self, name='macaroni_and_cheese', obj_name='macaroni_and_cheese'
    ):
        super().__init__(name, obj_name)
        self.rotation = {
            'x': (np.pi / 2, np.pi / 2),
            'z': (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None


@register_object
class Bagel(HopeBaseObject):
    def __init__(self, name='bagel', obj_name='bagel'):
        super().__init__(name, obj_name)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class Cake(HopeBaseObject):
    def __init__(self, name='cake', obj_name='cake'):
        super().__init__(name, obj_name)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class CakeN(HopeBaseObject):
    def __init__(self, name='cake_n', obj_name='cake_n'):
        super().__init__(name, obj_name)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class ChiffonCake(HopeBaseObject):
    def __init__(self, name='chiffon_cake', obj_name='chiffon_cake'):
        super().__init__(name, obj_name)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class Donut(HopeBaseObject):
    def __init__(self, name='donut', obj_name='donut'):
        super().__init__(name, obj_name)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class DonutN(HopeBaseObject):
    def __init__(self, name='donut_n', obj_name='donut_n'):
        super().__init__(name, obj_name)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class HotDog(HopeBaseObject):
    def __init__(self, name='hot_dog', obj_name='hot_dog'):
        super().__init__(name, obj_name)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class Mayo(HopeBaseObject):
    def __init__(self, name='mayo', obj_name='mayo'):
        super().__init__(name, obj_name)
        self.rotation = {
            'x': (np.pi / 2, np.pi / 2),
            'z': (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None


@register_object
class Milk(HopeBaseObject):
    def __init__(self, name='milk', obj_name='milk'):
        super().__init__(name, obj_name)
        self.rotation = {
            'x': (np.pi / 2, np.pi / 2),
            'z': (np.pi / 2, np.pi / 2),
        }


@register_object
class Egg(HopeBaseObject):
    def __init__(self, name='egg', obj_name='egg'):
        super().__init__(name, obj_name)


@register_object
class HotDogN(HopeBaseObject):
    def __init__(self, name='hot_dog_n', obj_name='hot_dog_n'):
        super().__init__(name, obj_name)


@register_object
class Chocolate(HopeBaseObject):
    def __init__(self, name='chocolate', obj_name='chocolate'):
        super().__init__(name, obj_name)


@register_object
class DumpTruck(HopeBaseObject):
    def __init__(self, name='dump_truck', obj_name='dump_truck'):
        super().__init__(name, obj_name)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'z'


@register_object
class Cereal(HopeBaseObject):
    def __init__(self, name='cereal', obj_name='cereal'):
        super().__init__(name, obj_name)
        self.rotation = (0.5, 0.5)
        self.rotation_axis = 'x'


# class Mushrooms(HopeBaseObject):
#     def __init__(self,
#                  name="mushrooms",
#                  obj_name="mushrooms"):
#         super().__init__(name, obj_name)

# class Mustard(HopeBaseObject):
#     def __init__(self,
#                  name="mustard",
#                  obj_name="mustard"):
#         super().__init__(name, obj_name)
#         self.rotation={
#             "x": (np.pi / 2, np.pi/2),
#             "z": (np.pi / 2, np.pi/2),
#         }
#         self.rotation_axis= None


@register_object
class OrangeJuice(HopeBaseObject):
    def __init__(self, name='orange_juice', obj_name='orange_juice'):
        super().__init__(name, obj_name)
        self.rotation = {
            'x': (np.pi / 2, np.pi / 2),
            'z': (np.pi / 2, np.pi / 2),
        }


# class Parmesan(HopeBaseObject):
#     def __init__(self,
#                  name="parmesan",
#                  obj_name="parmesan"):
#         super().__init__(name, obj_name)

# class Peaches(HopeBaseObject):
#     def __init__(self,
#                  name="peaches",
#                  obj_name="peaches"):
#         super().__init__(name, obj_name)

# class PeasAndCarrots(HopeBaseObject):
#     def __init__(self,
#                  name="peas_and_carrots",
#                  obj_name="peas_and_carrots"):
#         super().__init__(name, obj_name)

# class Pineapple(HopeBaseObject):
#     def __init__(self,
#                  name="pineapple",
#                  obj_name="pineapple"):
#         super().__init__(name, obj_name)


@register_object
class Popcorn(HopeBaseObject):
    def __init__(self, name='popcorn', obj_name='popcorn'):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = 'x'


@register_object
class Bread(HopeBaseObject):
    def __init__(self, name='bread', obj_name='bread'):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = 'x'


@register_object
class Steak(HopeBaseObject):
    def __init__(self, name='steak', obj_name='steak'):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = 'x'


# class Raisins(HopeBaseObject):
#     def __init__(self,
#                  name="raisins",
#                  obj_name="raisins"):
#         super().__init__(name, obj_name)


@register_object
class SaladDressing(HopeBaseObject):
    def __init__(self, name='salad_dressing', obj_name='salad_dressing'):
        super().__init__(name, obj_name)
        self.rotation = {
            'x': (np.pi / 2, np.pi / 2),
            'z': (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None


@register_object
class NewSaladDressing(HopeBaseObject):
    def __init__(
        self, name='new_salad_dressing', obj_name='new_salad_dressing'
    ):
        super().__init__(name, obj_name)
        self.rotation = {
            'x': (np.pi / 2, np.pi / 2),
            'z': (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None


@register_object
class TomatoSauce(HopeBaseObject):
    def __init__(self, name='tomato_sauce', obj_name='tomato_sauce'):
        super().__init__(name, obj_name)
        self.rotation_axis = 'z'


@register_object
class HammerHandle(HopeBaseObject):
    def __init__(
        self,
        name='hammer_handle',
        obj_name='hammer_handle',
        duplicate_collision_geoms=True,
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (0, 0)
        self.rotation_axis = 'x'


@register_object
class Hammer(HopeBaseObject):
    def __init__(
        self, name='hammer', obj_name='hammer', duplicate_collision_geoms=True
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class Knife(HopeBaseObject):
    def __init__(
        self, name='knife', obj_name='knife', duplicate_collision_geoms=True
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (-np.pi / 2, -np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class KnifeN(HopeBaseObject):
    def __init__(
        self,
        name='knife_n',
        obj_name='knife_n',
        duplicate_collision_geoms=True,
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (-np.pi / 2, -np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class RotatedKnife(HopeBaseObject):
    def __init__(
        self,
        name='rotated_knife',
        obj_name='knife_n',
        duplicate_collision_geoms=True,
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (-np.pi / 6, -np.pi / 6)
        self.rotation_axis = 'x'


@register_object
class RotatedKnifePi(HopeBaseObject):
    def __init__(
        self,
        name='rotated_knife_pi',
        obj_name='knife_n',
        duplicate_collision_geoms=True,
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'


# @register_object
# class KnifeN(HopeBaseObject):
#     def __init__(self, name="knife_n", obj_name="knife_n", duplicate_collision_geoms=True):
#         super().__init__(name, obj_name, duplicate_collision_geoms)
#         self.rotation = (0, 0)
#         self.rotation_axis = "x"

# class Tuna(HopeBaseObject):
#     def __init__(self,
#                  name="tuna",
#                  obj_name="tuna"):
#         super().__init__(name, obj_name)


# class Yogurt(HopeBaseObject):
#     def __init__(self,
#                  name="yogurt",
#                  obj_name="yogurt"):
#         super().__init__(name, obj_name)
@register_object
class Scissors(HopeBaseObject):
    def __init__(
        self,
        name='scissors',
        obj_name='scissors',
        duplicate_collision_geoms=True,
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (-np.pi / 2, -np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class RotatedScissorsPi(HopeBaseObject):
    def __init__(
        self,
        name='rotated_scissors_pi',
        obj_name='scissors',
        duplicate_collision_geoms=True,
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class RotatedScissors(HopeBaseObject):
    def __init__(
        self,
        name='rotated_scissors',
        obj_name='scissors',
        duplicate_collision_geoms=True,
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (-np.pi / 5, -np.pi / 5)
        self.rotation_axis = 'x'


@register_object
class ScissorsN(HopeBaseObject):
    def __init__(
        self,
        name='scissors_n',
        obj_name='scissors_n',
        duplicate_collision_geoms=True,
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (-np.pi / 2, -np.pi / 2)
        self.rotation_axis = 'z'


@register_object
class Fork(HopeBaseObject):
    def __init__(
        self, name='fork', obj_name='fork', duplicate_collision_geoms=True
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (-np.pi / 2, -np.pi / 2)
        self.rotation_axis = 'x'


@register_object
class RotatedFork(HopeBaseObject):
    def __init__(
        self,
        name='rotated_fork',
        obj_name='fork',
        duplicate_collision_geoms=True,
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (np.pi / 6, np.pi / 6)
        self.rotation_axis = 'x'


@register_object
class RotatedForkPi(HopeBaseObject):
    def __init__(
        self,
        name='rotated_fork_pi',
        obj_name='fork',
        duplicate_collision_geoms=True,
    ):
        super().__init__(name, obj_name, duplicate_collision_geoms)
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = 'x'
