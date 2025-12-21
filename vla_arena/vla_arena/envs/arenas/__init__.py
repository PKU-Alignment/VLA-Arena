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

from .arena import Arena
from .coffee_table_arena import CoffeeTableArena
from .empty_arena import EmptyArena
from .kitchen_arena import KitchenTableArena
from .living_room_arena import LivingRoomTableArena
from .study_arena import StudyTableArena
from .table_arena import TableArena


AGENTVIEW_CONFIG = {
    'floor': {
        'camera_name': 'agentview',
        'pos': [0.8965773716836134, 5.216182733499864e-07, 0.65],
        'quat': [
            0.6182166934013367,
            0.3432307541370392,
            0.3432314395904541,
            0.6182177066802979,
        ],
    },
    'main_table': {
        'camera_name': 'agentview',
        'pos': [0.6586131746834771, 0.0, 1.6103500240372423],
        'quat': [
            0.6380177736282349,
            0.3048497438430786,
            0.30484986305236816,
            0.6380177736282349,
        ],
    },
    'coffee_table': {
        'camera_name': 'agentview',
        'pos': [0.6586131746834771, 0.0, 1.6103500240372423],
        'quat': [
            0.6380177736282349,
            0.3048497438430786,
            0.30484986305236816,
            0.6380177736282349,
        ],
    },
    'kitchen_table': {
        'camera_name': 'agentview',
        'pos': [0.6586131746834771, 0.0, 1.6103500240372423],
        'quat': [
            0.6380177736282349,
            0.3048497438430786,
            0.30484986305236816,
            0.6380177736282349,
        ],
    },
    'living_room_table': {
        'camera_name': 'agentview',
        'pos': [0.6065773716836134, 0.0, 0.96],
        'quat': [
            0.6182166934013367,
            0.3432307541370392,
            0.3432314395904541,
            0.6182177066802979,
        ],
    },
    'study_table': {
        'camera_name': 'agentview',
        'pos': [0.4586131746834771, 0.0, 1.6103500240372423],
        'quat': [
            0.6380177736282349,
            0.3048497438430786,
            0.30484986305236816,
            0.6380177736282349,
        ],
    },
}
