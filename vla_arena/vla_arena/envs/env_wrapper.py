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

import matplotlib.cm as cm
import numpy as np
import robosuite as suite
from robosuite.utils.errors import RandomizationError

import vla_arena.vla_arena.envs.bddl_utils as BDDLUtils
from vla_arena.vla_arena.envs import *


class ControlEnv:
    def __init__(
        self,
        bddl_file_name,
        robots=['Panda'],
        controller='default_panda',
        gripper_types='default',
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera='frontview',
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            'agentview',
            'robot0_eye_in_hand',
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        renderer='mujoco',
        renderer_config=None,
        camera_offset=False,
        color_randomize=False,
        add_noise=False,
        light_adjustment=False,
        **kwargs,
    ):
        assert os.path.exists(
            bddl_file_name
        ), f'[error] {bddl_file_name} does not exist!'

        controller_configs = suite.load_composite_controller_config(
            robot=robots[0]
        )

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        # Check if we're using a multi-armed environment and use env_configuration argument if so

        # Create environment
        self.problem_name = problem_info['problem_name']
        self.domain_name = problem_info['domain_name']
        self.language_instruction = problem_info['language_instruction']
        self.env = TASK_MAPPING[self.problem_name](
            bddl_file_name,
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            camera_offset=camera_offset,
            color_randomize=color_randomize,
            add_noise=add_noise,
            light_adjustment=light_adjustment,
            **kwargs,
        )

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        success = False
        while not success:
            try:
                ret = self.env.reset()
                success = True
            except RandomizationError:
                pass
            finally:
                continue

        return ret

    def check_success(self):
        return self.env._check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        np.random.seed(seed)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self.env.close()
        del self.env

    def get_state(self):
        sim_state = self.env.sim.get_state()
        return {'qpos': sim_state.qpos.copy(), 'qvel': sim_state.qvel.copy()}


class OffScreenRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(
        self,
        camera_offset=False,
        color_randomize=False,
        add_noise=False,
        light_adjustment=False,
        **kwargs,
    ):
        # This shouldn't be customized
        kwargs['has_renderer'] = False
        kwargs['has_offscreen_renderer'] = True
        super().__init__(
            camera_offset=camera_offset,
            color_randomize=color_randomize,
            add_noise=add_noise,
            light_adjustment=light_adjustment,
            **kwargs,
        )




class DemoRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs['has_renderer'] = False
        kwargs['has_offscreen_renderer'] = True
        kwargs['render_camera'] = 'frontview'

        super().__init__(**kwargs)

    def _get_observations(self):
        return self.env._get_observations()
