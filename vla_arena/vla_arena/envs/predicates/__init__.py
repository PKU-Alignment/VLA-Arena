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

from .base_predicates import *


VALIDATE_PREDICATE_FN_DICT = {
    'true': TruePredicateFn(),
    'false': FalsePredicateFn(),
    'in': In(),
    'notin': NotIn(),
    'incontact': InContactPredicateFn(),
    'on': On(),
    'noton': NotOn(),
    'up': Up(),
    # "stack":     Stack(),
    # "temporal":  TemporalPredicate(),
    'printjointstate': PrintJointState(),
    'open': Open(),
    'close': Close(),
    'turnon': TurnOn(),
    'turnoff': TurnOff(),
    'collide': Collide(),
    'fall': Fall(),
    'checkforce': CheckForce(),
    'checkdistance': CheckDistance(),
    'incontactpart': InContactPart(),
    'checkgrippercontact': CheckGripperContact(),
    'checkgrippercontactpart': CheckGripperContactPart(),
    'checkgripperdistance': CheckGripperDistance(),
    'checkgripperdistancepart': CheckGripperDistancePart(),
}

TEMPORAL_PREDICATE_FN_LIST = [
    'incontact',
    'on',
    'up',
    'stack',
    'checkforce',
    'incontactpart',
    'checkdistance',
    'checkgrippercontact',
    'checkgrippercontactpart',
    'checkgripperdistance',
    'checkgripperdistancepart',
]


def update_predicate_fn_dict(fn_key, fn_name):
    VALIDATE_PREDICATE_FN_DICT.update({fn_key: eval(fn_name)()})


def eval_predicate_fn(predicate_fn_name, *args):
    assert predicate_fn_name in VALIDATE_PREDICATE_FN_DICT
    return VALIDATE_PREDICATE_FN_DICT[predicate_fn_name](*args)


def get_predicate_fn_dict():
    return VALIDATE_PREDICATE_FN_DICT


def get_predicate_fn(predicate_fn_name):
    return VALIDATE_PREDICATE_FN_DICT[predicate_fn_name.lower()]


def check_temporal_predicate(predicate_fn_name):
    return predicate_fn_name.lower() in TEMPORAL_PREDICATE_FN_LIST
