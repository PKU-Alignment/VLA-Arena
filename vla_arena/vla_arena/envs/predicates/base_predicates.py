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


class Expression:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class UnaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1):
        raise NotImplementedError


class BinaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1, arg2):
        raise NotImplementedError


class MultiarayAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, *args):
        raise NotImplementedError


class TruePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return True


class FalsePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return False


class InContactPredicateFn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg1.check_contact(arg2)


class In(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_contact(arg1) and arg2.check_contain(arg1)


class NotIn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return not arg2.check_contact(arg1) and not arg2.check_contain(arg1)


class On(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_ontop(arg1)

        # if arg2.object_state_type == "site":
        #     return arg2.check_ontop(arg1)
        # else:
        #     obj_1_pos = arg1.get_geom_state()["pos"]
        #     obj_2_pos = arg2.get_geom_state()["pos"]
        #     # arg1.on_top_of(arg2) ?
        #     # TODO (Yfeng): Add checking of center of mass are in the same regions
        #     if obj_1_pos[2] >= obj_2_pos[2] and arg2.check_contact(arg1):
        #         return True
        #     else:
        #         return False


class NotOn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return not arg2.check_ontop(arg1)


class Up(BinaryAtomic):
    def __call__(self, arg1):
        return arg1.get_geom_state()['pos'][2] >= 1.0


class Stack(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return (
            arg1.check_contact(arg2)
            and arg2.check_contain(arg1)
            and arg1.get_geom_state()['pos'][2]
            > arg2.get_geom_state()['pos'][2]
        )


class PrintJointState(UnaryAtomic):
    """This is a debug predicate to allow you print the joint values of the object you care"""

    def __call__(self, arg):
        print(arg.get_joint_state())
        return True


class Open(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_open()


class Close(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_close()


class TurnOn(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_on()


class TurnOff(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_off()


class Collide(UnaryAtomic):
    """Check if an object has been collided with."""

    def __call__(self, arg):
        return arg.check_collision()


class Fall(UnaryAtomic):
    def __call__(self, arg):
        return arg.fall()


class CheckForce(UnaryAtomic):
    def __call__(self, arg1, arg2):
        return arg1.check_force(arg2)


class CheckDistance(UnaryAtomic):
    def __call__(self, arg1, arg2):
        return arg1.check_distance(arg2)


class CheckGripperDistance(UnaryAtomic):
    def __call__(self, arg):
        return arg.check_gripper_distance()


class CheckGripperDistancePart(UnaryAtomic):
    def __call__(self, arg1, arg2):
        return arg1.check_gripper_distance_part(arg2)


class InContactPart(UnaryAtomic):
    def __call__(self, arg1, arg2, arg3, arg4):
        return arg1.check_in_contact_part(arg2, arg3, arg4)


class CheckGripperContact(UnaryAtomic):
    def __call__(self, arg1):
        return arg1.check_gripper_contact()


class CheckGripperContactPart(UnaryAtomic):
    def __call__(self, arg1, arg2):
        return arg1.check_gripper_contact_part(arg2)
