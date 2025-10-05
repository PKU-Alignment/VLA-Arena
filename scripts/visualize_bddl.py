import os
import argparse
import sys
from vla_arena.vla_arena.envs.env_wrapper import OffScreenRenderEnv
import math
import os

import imageio
import numpy as np
import tensorflow as tf
import time
import torch

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
def get_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def get_random_action():
    return [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1), -1]

def get_image(obs, cam_name):
    img = obs[cam_name+"_image"]
    img = img[::-1, ::-1, :]
    return img

def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

# Add parent directory to path for imports
sys.path.append("..")

def debug_single_file(bddl_file: str):
    print(f"Debugging file: {bddl_file}")
    resolution = 1024
    # 初始化并返回LIBERO环境
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    camera_name=env.env.camera_names[0]

    # 1. 加载环境并获取初始观测
    obs = env.reset()
    replay_images = [get_image(obs,camera_name)]

    # 2. 运行一段时间并收集图像
    t = 0
    cost = 0
    done = False
    while t < 100:
        obs, reward, done, info = env.step(get_random_action())
        # print(obs)
        cost += info.get('cost', 0)
        replay_images.append(get_image(obs,camera_name))
        t += 1
        print(f"Step {t}, cumulative cost: {cost}")
        if done:
            break

    # 3. 保存回放视频
    task_name = os.path.basename(bddl_file)
    save_rollout_video(
        replay_images,
        1,
        success=done,
        task_description=task_name,
        log_file=None
    )

    # 4. 关闭环境
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="递归查找并调试所有 .bddl 文件"
    )
    parser.add_argument(
        "--bddl_file",
        type=str,
        required=True,
        help="BDDL 文件路径或目录"
    )
    args = parser.parse_args()

    path = args.bddl_file
    if os.path.isfile(path):
        # 如果是文件，直接调试
        debug_single_file(path)

    elif os.path.isdir(path):
        # 递归遍历目录，查找所有 .bddl 文件
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.lower().endswith('.bddl'):
                    bddl_path = os.path.join(root, filename)
                    debug_single_file(bddl_path)

    else:
        print(f"错误: '{path}' 既不是文件也不是目录")


if __name__ == "__main__":
    main()
