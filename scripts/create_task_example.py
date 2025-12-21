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

"""This is a standalone file for create a task in vla arena."""

from vla_arena.vla_arena.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from vla_arena.vla_arena.utils.mu_utils import (
    InitialSceneTemplates,
    register_mu,
)
from vla_arena.vla_arena.utils.task_generation_utils import (
    generate_bddl_from_task_info,
    register_task_info,
)


@register_mu(scene_type='kitchen')
class KitchenScene1(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            'table': 1,
            'wooden_cabinet': 1,
            'flat_stove': 1,
        }

        object_num_info = {
            'akita_black_bowl': 6,
            'cookies': 1,
            'glazed_rim_porcelain_ramekin': 1,
            'plate': 1,
        }

        super().__init__(
            workspace_name='main_table',
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.12, -0.27],
                region_name='cabinet_region',
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(2.66, 2.72),
            ),
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.18, 0.14],
                region_name='plate_region',
                target_name=self.workspace_name,
                region_half_len=0.07,
            ),
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.01, 0.17],
                region_name='next_to_plate_region',
                target_name=self.workspace_name,
                region_half_len=0.15,
            ),
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, 0.2],
                region_name='ramekin_region',
                target_name=self.workspace_name,
                region_half_len=0.01,
            ),
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.03, 0.01],
                region_name='table_center',
                target_name=self.workspace_name,
                region_half_len=0.06,
            ),
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.41, -0.14],
                region_name='stove_region',
                target_name=self.workspace_name,
                region_half_len=0.01,
            ),
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, 0.20],
                region_name='between_plate_ramekin_region',
                target_name=self.workspace_name,
                region_half_len=0.01,
            ),
        )

        self.xy_region_kwargs_list = (
            get_xy_region_kwargs_list_from_regions_info(self.regions)
        )

    @property
    def init_states(self):
        return [
            (
                'On',
                'akita_black_bowl_1',
                'main_table_between_plate_ramekin_region',
            ),
            ('On', 'akita_black_bowl_2', 'glazed_rim_porcelain_ramekin_1'),
            ('On', 'plate_1', 'main_table_plate_region'),
            ('On', 'cookies_1', 'main_table_box_region'),
            (
                'On',
                'glazed_rim_porcelain_ramekin_1',
                'main_table_ramekin_region',
            ),
            ('On', 'wooden_cabinet_1', 'main_table_cabinet_region'),
            ('On', 'flat_stove_1', 'main_table_stove_region'),
            ('On', 'akita_black_bowl_3', 'akita_black_bowl_1'),
            ('On', 'akita_black_bowl_4', 'akita_black_bowl_3'),
            ('On', 'akita_black_bowl_5', 'akita_black_bowl_4'),
            ('On', 'akita_black_bowl_6', 'akita_black_bowl_5'),
        ]


def main():
    # kitchen_scene_1
    scene_name = 'kitchen_scene1'
    language = (
        'Pick the akita black bowl on the ramekin and place it on the plate'
    )
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=['wooden_cabinet_1', 'akita_black_bowl_1'],
        goal_states=[
            ('Open', 'wooden_cabinet_1_top_region'),
            ('In', 'akita_black_bowl_1', 'wooden_cabinet_1_top_region'),
        ],
    )

    scene_name = 'kitchen_scene1'
    language = 'Your Language 2'
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=['wooden_cabinet_1', 'akita_black_bowl_1'],
        goal_states=[
            ('Open', 'wooden_cabinet_1_top_region'),
            ('In', 'akita_black_bowl_1', 'wooden_cabinet_1_bottom_region'),
        ],
    )
    bddl_file_names, failures = generate_bddl_from_task_info()
    print(bddl_file_names)


if __name__ == '__main__':
    main()
