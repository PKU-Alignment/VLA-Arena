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


# All paths are relative to this file's location
_BENCHMARK_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_default_path_dict(custom_location=None):
    if custom_location is None:
        benchmark_root_path = _BENCHMARK_ROOT
    else:
        benchmark_root_path = custom_location

    # This is a default path for localizing all the default bddl files
    bddl_files_default_path = os.path.join(benchmark_root_path, './bddl_files')

    # This is a default path for localizing all the default bddl files
    init_states_default_path = os.path.join(benchmark_root_path, './init_files')

    # This is a default path for localizing all the default assets
    assets_default_path = os.path.join(benchmark_root_path, './assets')

    return {
        'benchmark_root': benchmark_root_path,
        'bddl_files': os.path.join(benchmark_root_path, 'bddl_files'),
        'init_states': os.path.join(benchmark_root_path, 'init_files'),
        'datasets': os.path.join(benchmark_root_path, '..', 'datasets'),
        'assets': os.path.join(benchmark_root_path, 'assets'),
    }


def get_vla_arena_path(query_key):
    """Get path for a VLA-Arena resource, resolved relative to this file."""
    paths = get_default_path_dict()

    if query_key not in paths:
        raise KeyError(f"Key '{query_key}' not found. Available keys: {list(paths.keys())}")

    return os.path.abspath(paths[query_key])
