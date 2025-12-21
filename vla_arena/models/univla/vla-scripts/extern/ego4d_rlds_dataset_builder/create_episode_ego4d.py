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
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments for the Ego4D data processing script."""
    parser = argparse.ArgumentParser(
        description='Process Ego4D data to create fake episodes.'
    )

    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help='Directory containing the source video clips',
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        required=True,
        help='Directory to save the processed episodes',
    )
    parser.add_argument(
        '--annotation_file',
        type=str,
        required=True,
        help='Path to the annotation JSON file',
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=96,
        help='Number of worker processes to use (default: 96)',
    )
    parser.add_argument(
        '--target_size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Target size for resizing images as "height width" (default: 224 224)',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify saved episodes by loading them after creation',
    )

    return parser.parse_args()


def center_crop_and_resize(image, target_size=(224, 224)):
    """
    Center crop and resize the input image while maintaining aspect ratio.

    Args:
        image (numpy.ndarray): Input image array with shape (H, W, C)
        target_size (tuple): Desired output size as (height, width)

    Returns:
        numpy.ndarray: Resized image array with shape (target_height, target_width, C)
    """
    height, width, _ = image.shape

    # Determine which dimension to crop (the longer side)
    if height < width:
        # Landscape image - crop width
        crop_size = height
        start_x = (width - crop_size) // 2
        start_y = 0
    else:
        # Portrait image - crop height
        crop_size = width
        start_x = 0
        start_y = (height - crop_size) // 2

    # Perform center crop
    cropped_image = image[
        start_y : start_y + crop_size, start_x : start_x + crop_size, :
    ]

    # Convert to PIL Image for high-quality resizing
    pil_image = Image.fromarray(cropped_image)
    resized_image = pil_image.resize(target_size, Image.BILINEAR)

    return np.array(resized_image)


def create_fake_episode(
    clip_dir, save_dir, annotation, target_size, verify=False
):
    """
    Create a fake episode from a video clip by processing all frames.

    Args:
        clip_dir (str): Path to directory containing clip frames
        save_dir (str): Directory to save the output episode
        annotation (list): List of annotation dictionaries
        target_size (tuple): Target size for frame resizing
        verify (bool): Whether to verify the saved episode

    Returns:
        None (saves episode to disk as .npy file)
    """
    episode_data = []
    clip_name = os.path.basename(clip_dir)
    video_name = os.path.basename(os.path.dirname(clip_dir))

    # Find matching annotation for this clip
    caption = None
    episode_id = None
    for anno in annotation:
        if (
            anno['video_name'] == video_name
            and anno['action_name'] == clip_name
        ):
            caption = anno['language'][5:]  # Remove first 5 characters '#C C '
            episode_id = anno['id']
            break

    if caption is None or episode_id is None:
        print(f'No matching annotation found for {video_name}/{clip_name}')
        return

    save_path = os.path.join(save_dir, f'episode_{episode_id}.npy')

    # Process each frame in the clip
    for frame_name in sorted(os.listdir(clip_dir)):
        frame_path = os.path.join(clip_dir, frame_name)
        try:
            frame = np.load(frame_path)
            frame = frame[:, :, ::-1]  # Convert BGR to RGB
            frame = center_crop_and_resize(frame, target_size)

            episode_data.append(
                {
                    'image': np.asarray(frame, dtype=np.uint8),
                    'wrist_image': np.asarray(
                        np.zeros([1, 1, 1]), dtype=np.uint8
                    ),
                    'state': np.asarray(np.zeros(7), dtype=np.float32),
                    'action': np.asarray(np.zeros(7), dtype=np.float32),
                    'language_instruction': caption,
                }
            )
        except Exception as e:
            print(f'Error processing frame {frame_path}: {e!s}')
            continue

    # Save the episode data
    np.save(save_path, episode_data)

    # Optional verification step
    if verify:
        try:
            loaded_data = np.load(save_path, allow_pickle=True)
            if len(loaded_data) == 0:
                print(f'Warning: Empty episode saved at {save_path}')
        except Exception as e:
            print(f'Failed to verify saved episode {episode_id}: {e!s}')


def process_video(
    video_dir, target_dir, annotation, target_size, verify=False
):
    """
    Process all clips within a single video directory.

    Args:
        video_dir (str): Path to video directory containing clips
        target_dir (str): Directory to save processed episodes
        annotation (list): List of annotation dictionaries
        target_size (tuple): Target size for frame resizing
        verify (bool): Whether to verify saved episodes
    """
    for clip_name in sorted(os.listdir(video_dir)):
        clip_dir = os.path.join(video_dir, clip_name)
        create_fake_episode(
            clip_dir=clip_dir,
            save_dir=target_dir,
            annotation=annotation,
            target_size=target_size,
            verify=verify,
        )


def main():
    args = parse_arguments()

    # Create target directory if it doesn't exist
    os.makedirs(args.target_dir, exist_ok=True)

    # Load annotation file
    print('Loading annotation file...')
    try:
        with open(args.annotation_file) as f:
            annotation = json.load(f)
    except Exception as e:
        print(f'Failed to load annotation file: {e!s}')
        return

    # Get list of video directories
    video_dirs = [
        os.path.join(args.source_dir, d)
        for d in sorted(os.listdir(args.source_dir))
        if os.path.isdir(os.path.join(args.source_dir, d))
    ]

    print(
        f'Processing {len(video_dirs)} videos using {args.processes} workers...'
    )

    # Process videos in parallel
    with Pool(processes=args.processes) as pool:
        process_func = partial(
            process_video,
            target_dir=args.target_dir,
            annotation=annotation,
            target_size=tuple(args.target_size),
            verify=args.verify,
        )

        # Process with progress bar
        results = list(
            tqdm(
                pool.imap_unordered(process_func, video_dirs),
                total=len(video_dirs),
                desc='Processing videos',
            )
        )

    print('Ego4D data processing completed successfully!')


if __name__ == '__main__':
    main()
