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

"""Compute normalization statistics for a config."""

import vla_arena.models.openpi.src.openpi.training.config as _config
import tyro

from vla_arena.models.openpi.src.openpi.workflow_utils import compute_and_save_norm_stats


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    output_path = compute_and_save_norm_stats(config, max_frames=max_frames)
    print(f'Writing stats to: {output_path}')


if __name__ == '__main__':
    tyro.cli(main)
