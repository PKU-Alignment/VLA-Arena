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

import re

from vla_arena.vla_arena.envs.base_object import (
    OBJECTS_DICT,
    VISUAL_CHANGE_OBJECTS_DICT,
)

from .articulated_objects import *
from .google_scanned_objects import *
from .hope_objects import *
from .site_object import SiteObject
from .target_zones import *
from .turbosquid_objects import *


def get_object_fn(category_name):
    return OBJECTS_DICT[category_name.lower()]


def get_object_dict():
    return OBJECTS_DICT
