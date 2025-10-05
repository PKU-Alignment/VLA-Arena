"""Utils for evaluating OpenVLA or fine-tuned OpenVLA policies."""

import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Sequence

import json_numpy
import numpy as np
import torch.nn as nn
import requests
import tensorflow as tf
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

# Apply JSON numpy patch for serialization
json_numpy.patch()

import sys
sys.path.insert(0, '/DATA/disk0/borong/openvla-oft')
# Initialize important constants
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224  # Standard image size expected by OpenVLA

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
constants = LIBERO_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `vla_arena/evaluation/openvla_utils.py`!")

def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # Read and update the config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "processing_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "processing_prismatic.OpenVLAForActionPrediction",
    }

    # Write back the updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "processing_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "processing_prismatic.OpenVLAForActionPrediction"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint versions of both
    modeling_prismatic.py and processing_prismatic.py:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    # Find current files
    curr_files = {"modeling_prismatic.py": None, "processing_prismatic.py": None}

    for root, _, files in os.walk("./vla_arena/evaluation/policy/prismatic_for_openvla/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict

def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.

    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """
    if model_is_on_hf_hub(checkpoint_path):
        # Download dataset stats directly from HF Hub
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )


# def get_noisy_action_projector(cfg: Any, llm_dim: int) -> NoisyActionProjector:
#     """
#     Get noisy action projector for diffusion-based action prediction.

#     Args:
#         cfg: Configuration object with model parameters
#         llm_dim: Dimension of the language model

#     Returns:
#         NoisyActionProjector: The initialized noisy action projector
#     """
#     # Initialize projector and move to device
#     noisy_action_projector = NoisyActionProjector(
#         llm_dim=llm_dim,
#     ).to(DEVICE)
#     noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(DEVICE)
#     noisy_action_projector.eval()

#     # Find and load checkpoint
#     checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "noisy_action_projector")
#     state_dict = load_component_state_dict(checkpoint_path)
#     noisy_action_projector.load_state_dict(state_dict)

#     return noisy_action_projector





def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    # print(f"image", img[0])
    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def prepare_images_for_vla(images: List[np.ndarray], center_crop: bool = True) -> List[Image.Image]:
    """
    Prepare images for VLA input by resizing and cropping as needed.

    Args:
        images: List of input images as numpy arrays
        center_crop: Whether to center crop the images

    Returns:
        List[Image.Image]: Processed images ready for the model
    """
    processed_images = []

    for image in images:
        # Validate format
        check_image_format(image)

        # Resize if needed
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

        # Convert to PIL image
        pil_image = Image.fromarray(image).convert("RGB")

        # Apply center crop if configured
        if center_crop:
            pil_image = center_crop_image(pil_image)

        processed_images.append(pil_image)

    return processed_images

def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]

def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict

def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio

def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image