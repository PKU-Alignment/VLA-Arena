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

from robosuite.utils.mjcf_utils import xml_path_completion

from vla_arena.vla_arena.envs.arenas import Arena
from vla_arena.vla_arena.envs.arenas.style import get_texture_filename


class EmptyArena(Arena):
    """Empty workspace."""

    def __init__(
        self,
        xml='arenas/empty_arena.xml',
        floor_style='light-gray',
        wall_style='light-gray-plaster',
    ):
        super().__init__(xml_path_completion(xml))
        self.floor_body = self.worldbody.find("./body[@name='floor']")

        texplane = self.asset.find("./texture[@name='texplane']")
        plane_file = texplane.get('file')
        plane_file = '/'.join(
            plane_file.split('/')[:-1]
            + [get_texture_filename(type='floor', style=floor_style)],
        )
        texplane.set('file', plane_file)

        texwall = self.asset.find("./texture[@name='tex-wall']")
        wall_file = texwall.get('file')
        wall_file = '/'.join(
            wall_file.split('/')[:-1]
            + [get_texture_filename(type='wall', style=wall_style)],
        )
        texwall.set('file', wall_file)
