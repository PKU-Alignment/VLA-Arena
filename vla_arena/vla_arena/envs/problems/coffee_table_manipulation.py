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

from robosuite.utils.mjcf_utils import new_site

from vla_arena.vla_arena.envs.arenas import AGENTVIEW_CONFIG
from vla_arena.vla_arena.envs.bddl_base_domain import (
    BDDLBaseDomain,
    register_problem,
)
from vla_arena.vla_arena.envs.objects import *
from vla_arena.vla_arena.envs.predicates import *
from vla_arena.vla_arena.envs.regions import *
from vla_arena.vla_arena.envs.robots import *


@register_problem
class Coffee_Table_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = 'coffee_table'
        self.visualization_sites_list = []
        if 'coffee_table_full_size' in kwargs:
            self.coffee_table_full_size = coffee_table_full_size
        else:
            self.coffee_table_full_size = (0.70, 1.6, 0.024)
        self.coffee_table_offset = (0, 0, 0.39)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.coffee_table_full_size[2]
        kwargs.update(
            {
                'robots': [
                    f'OnTheGround{robot_name}'
                    for robot_name in kwargs['robots']
                ]
            }
        )
        kwargs.update({'workspace_offset': self.coffee_table_offset})
        kwargs.update({'arena_type': 'coffee_table'})
        kwargs.update(
            {
                'scene_xml': 'scenes/coffee_table_blue_style.xml',
                'scene_properties': {
                    'floor_style': 'wood-plank',
                    'wall_style': 'light-gray-plaster',
                },
            },
        )

        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem['fixtures'].keys()):
            if fixture_category == 'coffee_table':
                continue

            for fixture_instance in self.parsed_problem['fixtures'][
                fixture_category
            ]:
                self.fixtures_dict[fixture_instance] = get_object_fn(
                    fixture_category
                )(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem['objects']
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem['regions']
        for object_region_name in list(region_dict.keys()):

            if 'coffee_table' in object_region_name:
                ranges = region_dict[object_region_name]['ranges'][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = (
                    (ranges[2] - ranges[0]) / 2,
                    (ranges[3] - ranges[1]) / 2,
                )
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2,
                    (ranges[3] + ranges[1]) / 2,
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]['rgba'],
                    zone_size=zone_size,
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone

                mujoco_arena.coffee_table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type='box',
                    ),
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for name, body in query_dict.items():
                    try:
                        if 'worldbody' not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find('body').findall('.//body'):
                        sites = part.findall('.//site')
                        joints = part.findall('./joint')
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get('name')
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = (
                                    SiteObject(
                                        name=site_name,
                                        parent_name=body.name,
                                        joints=[
                                            joint.get('name')
                                            for joint in joints
                                        ],
                                        size=site.get('size'),
                                        rgba=site.get('rgba'),
                                        site_type=site.get('type'),
                                        site_pos=site.get('pos'),
                                        site_quat=site.get('quat'),
                                        object_properties=body.object_properties,
                                    )
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties['vis_site_names'] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name)
                .object_properties['vis_site_names']
                .items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if (
                    (self.sim.model.site_rgba[vis_g_id][3] <= 0)
                    and site_visible
                ) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0)
                    and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena, camera_names, camera_configs):
        for camera in camera_names:
            if camera == 'robot0_eye_in_hand':
                continue
            if camera == 'agentview':
                mujoco_arena.set_camera(
                    **AGENTVIEW_CONFIG[self.workspace_name],
                    pos_offset=camera_configs[camera],
                )
            else:
                mujoco_arena.set_camera(
                    camera_name=camera, pos_offset=camera_configs[camera]
                )
        mujoco_arena.set_camera(
            camera_name='galleryview',
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )

        # robosuite's default agentview camera configuration
        mujoco_arena.set_camera(
            camera_name='canonical_agentview',
            pos=[0.5386131746834771, 0.0, 0.7903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )
