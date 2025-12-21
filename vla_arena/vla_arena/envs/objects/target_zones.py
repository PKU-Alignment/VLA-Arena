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

import pathlib
import re

import numpy as np


absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()
from robosuite.utils.mjcf_utils import (
    array_to_string,
)

from vla_arena.vla_arena.envs.base_object import register_object

# from robosuite.models.objects import BoxObject
from vla_arena.vla_arena.envs.objects.site_object import SiteObject


@register_object
class TargetZone(SiteObject):
    def __init__(
        self,
        name,
        zone_height=0.007,
        z_offset=0.02,
        rgba=(1, 0, 0, 1),
        joints=None,
        zone_size=(0.15, 0.05),
        zone_centroid_xy=(0, 0),
        # site_type="box",
        # site_pos="0 0 0",
        # site_quat="1 0 0 0",
    ):
        self.category_name = '_'.join(
            re.sub(r'([A-Z])', r' \1', self.__class__.__name__).split(),
        ).lower()
        self.size = (zone_size[0], zone_size[1], zone_height)
        self.pos = zone_centroid_xy + (z_offset,)
        self.quat = (1, 0, 0, 0)
        super().__init__(
            name=name,
            size=self.size,
            rgba=rgba,
            site_type='box',
            site_pos=array_to_string(self.pos),
            site_quat=array_to_string(self.quat),
        )

    def in_box(self, this_position, this_mat, other_position):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """

        total_size = np.abs(this_mat @ self.size)

        ub = this_position + total_size
        lb = this_position - total_size

        lb[2] -= 0.01
        return np.all(other_position > lb) and np.all(other_position < ub)

    def on_top(self, this_position, this_mat, other_position):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """

        # (TODO) Yifeng: The transformation for size is a little bit
        # hacky at the moment. Will dig deeper into it.
        total_size = np.abs(this_mat @ self.size)
        ub = this_position + total_size
        return np.all(other_position > ub)
