import argparse
import os
from pathlib import Path
import h5py
import numpy as np
import json
import glob
import robosuite
import robosuite.utils.transform_utils as T
import robosuite.macros as macros

import init_path
import vla_arena.vla_arena.utils.utils as vla_arena_utils
import cv2
from PIL import Image
from robosuite.utils import camera_utils

from vla_arena.vla_arena.envs import *
from vla_arena.vla_arena import get_vla_arena_path


def process_single_demo_file(demo_file_path, env_kwargs_template, args, global_demo_counter):
    """
    处理单个demo HDF5文件并返回处理后的数据
    
    Args:
        demo_file_path: 原始demo文件路径
        env_kwargs_template: 环境参数模板
        args: 命令行参数
        global_demo_counter: 全局demo计数器
        
    Returns:
        处理后的demo数据列表和更新后的计数器
    """
    
    print(f"\n处理文件: {demo_file_path}")
    
    try:
        f = h5py.File(demo_file_path, "r")
    except Exception as e:
        print(f"无法打开文件 {demo_file_path}: {e}")
        return [], global_demo_counter
    
    # 提取必要的元数据
    try:
        env_name = f["data"].attrs["env"]
        env_info = f["data"].attrs["env_info"]
        problem_info = json.loads(f["data"].attrs["problem_info"])
        problem_name = problem_info["problem_name"]
        language_instruction = problem_info["language_instruction"]
        bddl_file_name = f["data"].attrs["bddl_file_name"]
        demos = list(f["data"].keys())
    except KeyError as e:
        print(f"文件 {demo_file_path} 缺少必要的元数据: {e}")
        f.close()
        return [], global_demo_counter
    
    # 更新环境参数
    env_kwargs = json.loads(env_info)
    vla_arena_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=not args.not_use_camera_obs,
        has_offscreen_renderer=not args.not_use_camera_obs,
        ignore_done=True,
        use_camera_obs=not args.not_use_camera_obs,
        camera_depths=args.use_depth,
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        camera_segmentations=None,
    )
    
    # 创建环境
    try:
        env = TASK_MAPPING[problem_name](**env_kwargs)
    except Exception as e:
        print(f"无法创建环境 {problem_name}: {e}")
        f.close()
        return [], global_demo_counter
    
    processed_demos = []
    cap_index = 5
    
    # 处理每个episode
    for ep in demos:
        print(f"  处理 {ep}...")
        
        try:
            # 读取模型和状态
            model_xml = f["data/{}".format(ep)].attrs["model_file"]
            states = f["data/{}/states".format(ep)][()]
            actions = np.array(f["data/{}/actions".format(ep)][()])
            
            # 重置环境
            reset_success = False
            max_reset_attempts = 5
            for attempt in range(max_reset_attempts):
                try:
                    env.reset()
                    reset_success = True
                    break
                except:
                    if attempt == max_reset_attempts - 1:
                        print(f"    无法重置环境，跳过 {ep}")
                    continue
            
            if not reset_success:
                continue
                
            model_xml = vla_arena_utils.postprocess_model_xml(model_xml, {})
            
            # 初始化环境状态
            init_idx = 0
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(states[init_idx])
            env.sim.forward()
            model_xml = env.sim.model.get_xml()

            camera_names = env.camera_names
            
            # 收集数据的容器
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            camera_list={}
            for camera in camera_names:
                camera_list[camera]={
                    "images": [],
                    "depths": [],
                }
            valid_index = []
            
            # 回放动作并收集观测
            for j, action in enumerate(actions):
                obs, reward, done, info = env.step(action)
                
                # 检查状态一致性
                if j < len(actions) - 1:
                    state_playback = env.sim.get_state().flatten()
                    err = np.linalg.norm(states[j + 1] - state_playback)
                    # if err > 0.01:
                    #     print(f"    [警告] 回放偏差 {err:.2f} at step {j}")
                
                # 跳过前几帧（传感器稳定）
                if j < cap_index:
                    continue
                    
                valid_index.append(j)
                
                # 收集proprioception数据
                if not args.no_proprio:
                    if "robot0_gripper_qpos" in obs:
                        gripper_states.append(obs["robot0_gripper_qpos"])
                    joint_states.append(obs["robot0_joint_pos"])
                    ee_states.append(
                        np.hstack((
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        ))
                    )
                
                robot_states.append(env.get_robot_state_vector(obs))
                
                # 收集图像数据
                if not args.not_use_camera_obs:
                    if args.use_depth:
                        for camera in camera_names:
                            camera_list[camera]["depths"].append(obs[camera+"_depth"])
                    for camera in camera_names:
                        camera_list[camera]["images"].append(obs[camera+"_image"])
            
            # 准备最终数据
            states = states[valid_index]
            actions = actions[valid_index]
            dones = np.zeros(len(actions)).astype(np.uint8)
            dones[-1] = 1
            rewards = np.zeros(len(actions)).astype(np.uint8)
            rewards[-1] = 1
            
            # 存储处理后的数据
            demo_data = {
                'demo_id': f"demo_{global_demo_counter}",
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
                'robot_states': np.stack(robot_states, axis=0) if robot_states else None,
                'model_file': model_xml,
                'init_state': states[init_idx] if len(states) > 0 else None,
                'num_samples': len(camera_list[camera_names[0]]["images"]),
                'source_file': demo_file_path,
                'original_ep': ep,
            }
            
            # 添加观测数据
            if not args.no_proprio and gripper_states:
                demo_data['gripper_states'] = np.stack(gripper_states, axis=0)
                demo_data['joint_states'] = np.stack(joint_states, axis=0)
                demo_data['ee_states'] = np.stack(ee_states, axis=0)
                demo_data['ee_pos'] = demo_data['ee_states'][:, :3]
                demo_data['ee_ori'] = demo_data['ee_states'][:, 3:]
            
            if not args.not_use_camera_obs:
                for camera in camera_names:
                    if camera_list[camera]["images"]:
                        demo_data[camera+"_rgb"] = np.stack(camera_list[camera]["images"], axis=0)
                
                if args.use_depth:
                    for camera in camera_names:
                        if camera_list[camera]["depths"]:
                            demo_data[camera+"_depth"] = np.stack(camera_list[camera]["depths"], axis=0)
            
            processed_demos.append(demo_data)
            global_demo_counter += 1
            
        except Exception as e:
            print(f"    处理 {ep} 时出错: {e}")
            continue
    
    # 清理
    env.close()
    f.close()
    
    # 返回元数据和处理后的demos
    metadata = {
        'env_name': env_name,
        'problem_info': problem_info,
        'bddl_file_name': bddl_file_name,
        'env_kwargs': env_kwargs,
        'camera_names': camera_names,
    }
    
    return processed_demos, global_demo_counter, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="包含原始demo HDF5文件的目录 (如: demonstration_data/xxx/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认根据BDDL文件自动确定"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.hdf5",
        help="要处理的文件名 (默认: .hdf5)"
    )
    parser.add_argument("--not-use-camera-obs", action="store_true")
    parser.add_argument("--no-proprio", action="store_true")
    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument(
        "--not-recursive",
        action="store_true",
        help="不递归搜索子目录"
    )
    
    args = parser.parse_args()
    
    # 查找所有要处理的HDF5文件
    if not args.not_recursive:
        demo_files = list(Path(args.input_dir).rglob(args.pattern))
    else:
        demo_files = list(Path(args.input_dir).glob(args.pattern))
    
    if not demo_files:
        print(f"在 {args.input_dir} 中没有找到匹配 {args.pattern} 的文件")
        return
    
    print(f"找到 {len(demo_files)} 个文件待处理")
    
    # 处理所有文件并收集数据，按BDDL文件分组
    demos_by_bddl = {}  # {bddl_file_name: [demos]}
    env_kwargs_template = {}
    metadata_by_bddl = {}  # {bddl_file_name: metadata}
    
    for demo_file in demo_files:
        demos, _, metadata = process_single_demo_file(
            str(demo_file), 
            env_kwargs_template, 
            args, 
            0  # 每个BDDL文件独立计数
        )
        
        if metadata and demos:
            bddl_file_name = metadata['bddl_file_name']
            if bddl_file_name not in demos_by_bddl:
                demos_by_bddl[bddl_file_name] = []
                metadata_by_bddl[bddl_file_name] = metadata
            demos_by_bddl[bddl_file_name].extend(demos)
    
    # 为每个BDDL文件创建一个输出文件
    for bddl_file_name, demos in demos_by_bddl.items():
        # 根据原代码的命名逻辑生成输出路径
        demo_dir = args.input_dir  # 输入目录作为demo_dir
        bddl_base_name = os.path.basename(bddl_file_name)
        
        if args.output_dir:
            # 如果指定了输出目录，使用它
            output_parent_dir = Path(args.output_dir)
            hdf5_file_name = bddl_base_name.replace(".bddl", "_demo.hdf5")
            hdf5_path = output_parent_dir / hdf5_file_name
        else:
            # 否则按原代码逻辑：基于demonstration_data目录结构
            if "demonstration_data/" in demo_dir:
                relative_dir = demo_dir.split("demonstration_data/")[-1]
            else:
                # 如果路径中没有demonstration_data，使用当前目录名
                relative_dir = os.path.basename(demo_dir)
            
            hdf5_file_name = bddl_base_name.replace(".bddl", "_demo.hdf5")
            hdf5_path = os.path.join(
                get_vla_arena_path("datasets"), 
                relative_dir, 
                hdf5_file_name
            )
            hdf5_path = Path(hdf5_path)
            if hdf5_path.exists():
                stem = hdf5_path.stem
                suffix = hdf5_path.suffix
                new_file_name = f"{stem}_1{suffix}"
                hdf5_path = hdf5_path.parent / new_file_name
        
        output_parent_dir = hdf5_path.parent
        output_parent_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n为 {bddl_base_name} 创建输出文件: {hdf5_path}")
        
        # 写入HDF5文件（使用原代码的结构）
        metadata = metadata_by_bddl[bddl_file_name]
        
        with h5py.File(str(hdf5_path), "w") as h5py_f:
            grp = h5py_f.create_group("data")
            
            # 写入属性（与原代码保持一致）
            grp.attrs["env_name"] = metadata['env_name']
            grp.attrs["problem_info"] = json.dumps(metadata['problem_info'])
            grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION
            
            # 环境参数
            problem_name = metadata['problem_info']['problem_name']
            env_args = {
                "type": 1,
                "env_name": metadata['env_name'],
                "problem_name": problem_name,
                "bddl_file": bddl_file_name,
                "env_kwargs": metadata['env_kwargs'],
            }
            grp.attrs["env_args"] = json.dumps(env_args)
            grp.attrs['camera_names'] = metadata['camera_names']
            
            grp.attrs["bddl_file_name"] = bddl_file_name
            if os.path.exists(bddl_file_name):
                grp.attrs["bddl_file_content"] = open(bddl_file_name, "r").read()
            
            # 写入每个demo的数据，重新编号
            total_len = 0
            for i, demo_data in enumerate(demos):
                demo_id = f"demo_{i}"  # 重新编号从0开始
                ep_data_grp = grp.create_group(demo_id)
                
                # 写入观测数据组
                obs_grp = ep_data_grp.create_group("obs")
                
                # Proprioception数据
                for key in ['gripper_states', 'joint_states', 'ee_states', 'ee_pos', 'ee_ori']:
                    if key in demo_data:
                        obs_grp.create_dataset(key, data=demo_data[key])
                
                # 图像数据
                for camera in metadata['camera_names']:
                    for key in [camera + suffix for suffix in ["_rgb", "_depth"]]:
                        if key in demo_data:
                            obs_grp.create_dataset(key, data=demo_data[key])
                
                # 写入动作和状态数据
                ep_data_grp.create_dataset("actions", data=demo_data['actions'])
                ep_data_grp.create_dataset("states", data=demo_data['states'])
                ep_data_grp.create_dataset("rewards", data=demo_data['rewards'])
                ep_data_grp.create_dataset("dones", data=demo_data['dones'])
                
                if demo_data['robot_states'] is not None:
                    ep_data_grp.create_dataset("robot_states", data=demo_data['robot_states'])
                
                # 写入属性
                ep_data_grp.attrs["num_samples"] = demo_data['num_samples']
                ep_data_grp.attrs["model_file"] = demo_data['model_file']
                if demo_data['init_state'] is not None:
                    ep_data_grp.attrs["init_state"] = demo_data['init_state']
                
                total_len += demo_data['num_samples']
            
            # 写入汇总信息
            grp.attrs["num_demos"] = len(demos)
            grp.attrs["total"] = total_len
        
        print(f"创建的数据集已保存到: {hdf5_path}")
        print(f"Demonstrations数: {len(demos)}")
        print(f"总样本数: {total_len}")


if __name__ == "__main__":
    main()