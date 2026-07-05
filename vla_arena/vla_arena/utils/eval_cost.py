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


def get_timeout_final_cost(env):
    if hasattr(env, 'get_final_cost'):
        return env.get_final_cost()

    inner_env = getattr(env, 'env', None)
    if inner_env is not None and hasattr(inner_env, 'get_final_cost'):
        return inner_env.get_final_cost()

    return 0


def is_success_done(done, info):
    return bool(info.get('success', done))
