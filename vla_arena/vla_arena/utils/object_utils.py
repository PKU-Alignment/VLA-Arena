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

# This is a util file for various functions that retrieve object information

from vla_arena.vla_arena.envs.objects import get_object_fn


EXCEPTION_DICT = {'flat_stove': 'flat_stove_burner'}


def update_exception_dict(object_name, site_name):
    """Update EXCEPTION_DICT information. This is to handle some special case of affordance region naming.

    Args:
        object_name (str): object name
        site_name (str): site name
    """
    EXCEPTION_DICT[object_name] = site_name


def get_affordance_regions(objects, verbose=False):
    """_summary_

    Args:
        objects (MujocoObject): a dictionary of objects
        verbose (bool, optional): Print additional debug information. Defaults to False.

    Returns:
        dict: a dictionary of object names and their affordance regions.
    """
    affordances = {}
    for object_name in objects.keys():
        try:
            obj = get_object_fn(object_name)()
            # print(obj.root.findall(".//site"))
            object_affordance = []
            for site in obj.root.findall('.//site'):
                site_name = site.get('name')
                if 'site' not in site_name and (
                    object_name not in EXCEPTION_DICT
                    or (
                        object_name in EXCEPTION_DICT
                        and site_name not in EXCEPTION_DICT[object_name]
                    )
                ):
                    # print(site_name)
                    # object name is already added as prefix when the object is initialized. remove them for consistency in bddl files
                    object_affordance.append(
                        site_name.replace(f'{object_name}_', '')
                    )
            if len(object_affordance) > 0:
                affordances[object_name] = object_affordance
        except:
            if verbose:
                print(f'Skipping {object_name}')

    return affordances
