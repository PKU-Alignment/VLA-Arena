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

import argparse
import json
import os
from multiprocessing import Manager, Pool

import cv2
import numpy as np
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments for video processing configuration."""
    parser = argparse.ArgumentParser(
        description='Process Ego4D video clips into frame sequences.'
    )

    # Required paths
    parser.add_argument(
        '--denseclips_dir',
        type=str,
        required=True,
        help='Root directory for denseclips output',
    )
    parser.add_argument(
        '--info_clips_json',
        type=str,
        required=True,
        help='Path to info_clips.json containing clip information',
    )
    parser.add_argument(
        '--source_videos_dir',
        type=str,
        required=True,
        help='Directory containing source video files',
    )

    # Processing options
    parser.add_argument(
        '--frame_interval',
        type=int,
        default=15,
        help='Interval between saved frames (default: 15)',
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=1,
        help='Number of parallel processes to use (default: 1)',
    )

    return parser.parse_args()


def read_frames_from_video(video_path):
    """Read all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def process_video(video_name, clips, args, info):
    """Process a single video and its clips, saving frames as numpy arrays."""
    video_path = os.path.join(args.source_videos_dir, f'{video_name}.mp4')
    frames = read_frames_from_video(video_path)

    for idx, clip in enumerate(clips):
        action_name = clip['pre_frame']['path'].split('/')[1]
        save_path = os.path.join(args.denseclips_dir, video_name, action_name)
        os.makedirs(save_path, exist_ok=True)

        start = clip['pre_frame']['frame_num']
        end = clip['post_frame']['frame_num']
        clip_frames = frames[start : end + 1]

        # Save frames at specified intervals
        for frame_count, frame in enumerate(clip_frames):
            if (
                frame_count % args.frame_interval == 0
                or frame_count == end - start
            ):
                npy_name = os.path.join(
                    save_path,
                    f'{frame_count//args.frame_interval + 1:05d}.npy',
                )
                if not os.path.exists(npy_name):
                    np.save(npy_name, frame)

        # Store annotation info
        info.append(
            {
                'video_name': video_name,
                'action_name': action_name,
                'source_video': video_path,
                'start_frame': start,
                'end_frame': end,
                'language': clip['narration_text'],
                'id': idx,
            }
        )


def main():
    args = parse_arguments()
    os.makedirs(args.denseclips_dir, exist_ok=True)

    with open(args.info_clips_json) as file:
        clip_data = json.load(file)

    if args.processes > 1:
        manager = Manager()
        info = manager.list()

        with Pool(processes=args.processes) as pool:
            pool.starmap(
                process_video,
                [
                    (video_name, clips, args, info)
                    for video_name, clips in clip_data.items()
                ],
            )
    else:
        info = []
        for video_name, clips in tqdm(
            clip_data.items(), desc='Processing videos'
        ):
            process_video(video_name, clips, args, info)

    # Save annotations
    with open(os.path.join(args.denseclips_dir, 'annotations.json'), 'w') as f:
        json.dump(list(info), f, indent=4)


if __name__ == '__main__':
    main()
