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


OBJECTS_DICT = {}
VISUAL_CHANGE_OBJECTS_DICT = {}


def register_object(target_class):
    """We design the mapping to be case-INsensitive."""
    key = '_'.join(
        re.sub(r'([A-Z0-9])', r' \1', target_class.__name__).split()
    ).lower()
    assert key not in OBJECTS_DICT
    OBJECTS_DICT[key] = target_class
    return target_class


def register_visual_change_object(target_class):
    """We keep track of objects that might have visual changes to optimize the codebase"""
    key = '_'.join(
        re.sub(r'([A-Z0-9])', r' \1', target_class.__name__).split()
    ).lower()
    VISUAL_CHANGE_OBJECTS_DICT[key] = target_class
    return target_class
