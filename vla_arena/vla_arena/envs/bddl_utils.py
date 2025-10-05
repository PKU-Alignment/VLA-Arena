from bddl.parsing import *

import itertools
import numpy as np

pi = np.pi


def get_regions(t, regions, group):
    group.pop(0)
    while group:
        region = group.pop(0)
        region_name = region[0]
        target_name = None
        region_dict = {
            "target": None,
            "ranges": [],
            "extra": [],
            "yaw_rotation": [0, 0],
            "rgba": [0, 0, 1, 0],
        }
        for attribute in region[1:]:
            if attribute[0] == ":target":
                assert len(attribute) == 2
                region_dict["target"] = attribute[1]
                target_name = attribute[1]
            elif attribute[0] == ":ranges":
                for rect_range in attribute[1]:
                    assert (
                        len(rect_range) == 4
                    ), f"Dimension of rectangular range mismatched!!, supposed to be 4, only found {len(rect_range)}"
                    region_dict["ranges"].append([float(x) for x in rect_range])
            elif attribute[0] == ":yaw_rotation":
                # print(attribute[1])
                for value in attribute[1]:
                    region_dict["yaw_rotation"] = [eval(x) for x in value]
            elif attribute[0] == ":rgba":
                assert (
                    len(attribute[1]) == 4
                ), f"Missing specification for rgba color, supposed to be 4 dimension, but only got  {attribute[1]}"
                region_dict["rgba"] = [float(x) for x in attribute[1]]
            else:
                raise NotImplementedError
        regions[target_name + "_" + region_name] = region_dict

def get_moving_objects(t, moving_objects, group):
    group.pop(0)
    while group:
        moving_object = group.pop(0)
        moving_object_dict = {}
        moving_object_dict["name"] = moving_object[0]
        for attribute in moving_object[1:]:
            if attribute[0] == ":motion_type":
                moving_object_dict["motion_type"] = attribute[1]
            elif attribute[0] == ":motion_speed":
                moving_object_dict["motion_speed"] = attribute[1]
            elif attribute[0] == ":motion_radius":
                moving_object_dict["motion_radius"] = attribute[1]
            elif attribute[0] == ":motion_center":
                moving_object_dict["motion_center"] = attribute[1]
            elif attribute[0] == ":motion_quat":
                moving_object_dict["motion_quat"] = attribute[1]
            elif attribute[0] == ":motion_pos":
                moving_object_dict["motion_pos"] = attribute[1]
            elif attribute[0] == ":motion_direction":
                moving_object_dict["motion_direction"] = attribute[1]
            elif attribute[0] == ":motion_angle":
                moving_object_dict["motion_angle"] = attribute[1]
            elif attribute[0] == ":motion_period":
                moving_object_dict["motion_period"] = attribute[1]
            elif attribute[0] == ":motion_travel_dist":
                moving_object_dict["motion_travel_dist"] = attribute[1]
            elif attribute[0] == ":motion_waypoints":
                moving_object_dict["motion_waypoints"] = attribute[1]
            elif attribute[0] == ":motion_dt":
                moving_object_dict["motion_dt"] = attribute[1]
            elif attribute[0] == ":motion_loop":
                moving_object_dict["motion_loop"] = attribute[1]
            elif attribute[0] == ":motion_initial_speed":
                moving_object_dict["motion_initial_speed"] = attribute[1]
            elif attribute[0] == ":motion_direction":
                moving_object_dict["motion_direction"] = attribute[1]
            elif attribute[0] == ":motion_start_pos":
                moving_object_dict["motion_start_pos"] = attribute[1]
            elif attribute[0] == ":motion_start_quat":
                moving_object_dict["motion_start_quat"] = attribute[1]
            elif attribute[0] == ":motion_center":
                moving_object_dict["motion_center"] = attribute[1]
            elif attribute[0] == ":motion_radius":
                moving_object_dict["motion_radius"] = attribute[1]
            elif attribute[0] == ":motion_gravity":
                moving_object_dict["motion_gravity"] = attribute[1]
            else:
                raise NotImplementedError(f"Invalid motion attribute: {attribute[0]}")
        moving_objects.append(moving_object_dict)

def get_scenes(t, scene_properties, group):
    group.pop(0)
    while group:
        scene_property = group.pop(0)
        scene_properties_dict = {}
        for attribute in region[1:]:
            if attribute[0] == ":floor":
                assert len(attribute) == 2
                scene_properties_dict["floor_style"] = attribute[1]
            elif attribute[0] == ":wall":
                assert len(attribute) == 2
                scene_properties_dict["wall_style"] = attribute[1]
            else:
                raise NotImplementedError


def get_problem_info(problem_filename):
    domain_name = "unknown"
    problem_filename = problem_filename
    tokens = scan_tokens(filename=problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        language_instruction = ""
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                domain_name = "robosuite"
            elif t == ":language":
                group.pop(0)
                language_instruction = group
    return {
        "problem_name": problem_name,
        "domain_name": domain_name,
        "language_instruction": " ".join(language_instruction),
    }


def robosuite_parse_problem(problem_filename):
    domain_name = "robosuite"
    problem_filename = problem_filename
    tokens = scan_tokens(filename=problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        objects = {}
        obj_of_interest = []
        initial_state = []
        goal_state = []
        fixtures = {}
        regions = {}
        image_settings={}
        scene_properties = {}
        language_instruction = ""
        cost_state = []
        moving_objects = []
        camera_names=[]
        noise=[]
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                if domain_name != group[-1]:
                    raise Exception("Different domain specified in problem file")
            elif t == ":requirements":
                pass
            elif t == ":objects":
                group.pop(0)
                object_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        objects[group.pop(0)] = object_list
                        object_list = []
                    else:
                        object_list.append(group.pop(0))
                if object_list:
                    if not "object" in objects:
                        objects["object"] = []
                    objects["object"] += object_list
            elif t == ":obj_of_interest":
                group.pop(0)
                while group:
                    obj_of_interest.append(group.pop(0))
            elif t == ":fixtures":
                group.pop(0)
                fixture_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        fixtures[group.pop(0)] = fixture_list
                        fixture_list = []
                    else:
                        fixture_list.append(group.pop(0))
                if fixture_list:
                    if not "fixture" in fixtures:
                        fixtures["fixture"] = []
                    fixtures["fixture"] += fixture_list
            elif t == ":regions":
                get_regions(t, regions, group)
            elif t == ":scene_properties":
                get_scenes(t, scene_properties, group)
            elif t == ":language":
                group.pop(0)
                language_instruction = group
            elif t == ":init":
                group.pop(0)
                initial_state = group
            elif t == ":goal":
                package_predicates(group[1], goal_state, "", "goals")
            elif t == ":cost":
                print(f"cost_state: {group[1]}")
                package_predicates(group[1], cost_state, "", "costs")
            elif t == ":moving_objects":
                get_moving_objects(t, moving_objects, group)
            elif t == ":image_settings":
                group.pop(0)
                while group:
                    if group[0].isalpha():
                        image_settings[group.pop(0)]=float(group.pop(1))
            elif t == ":camera":
                group.pop(0)
                while group:
                    camera_names.append(group.pop(0))
            elif t == ":noise":
                group.pop(0)
                if group[0]=="gaussian":
                    noise.append(group.pop(0))
                    noise.append(float(group.pop(0)))
                    noise.append(float(group.pop(0)))
                elif group[0]=="salt_pepper":
                    noise.append(group.pop(0))
                    noise.append(float(group.pop(0)))
            else:
                print("%s is not recognized in problem" % t)
        return {
            "problem_name": problem_name,
            "fixtures": fixtures,
            "regions": regions,
            "objects": objects,
            "scene_properties": scene_properties,
            "initial_state": initial_state,
            "goal_state": goal_state,
            "language_instruction": language_instruction,
            "obj_of_interest": obj_of_interest,
            "cost_state": cost_state,
            "moving_objects": moving_objects,
            "image_settings": image_settings,
            "camera_names": camera_names,
            "noise": noise,
        }
    else:
        raise Exception(
            f"Problem {behavior_activity} {activity_definition} does not match problem pattern"
        )
