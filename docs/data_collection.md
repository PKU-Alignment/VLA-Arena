# Data Collection Guide

VLA-Arena provides a complete framework for collecting data in custom scenes and converting the collected data format. This guide will help you understand how to collect data in existing scenes and convert it into a usable dataset.

## Table of Contents
1. [Collect Demonstration Data](#1-collect-demonstration-data)
2. [Convert Data Format](#2-convert-data-format)
3. [Regenerate Dataset](#3-regenerate-dataset)
4. [Convert Dataset to RLDS Format](#4-convert-dataset-to-rlds-format)
5. [Convert RLDS Dataset to LeRobot Format](#5-convert-rlds-dataset-to-lerobot-format)

## 1. Collect Demonstration Data
You can use `scripts/collect_demonstration.py` to collect demonstration data in simulation environment:
```bash
python scripts/collect_demonstration.py --bddl-file <your_bddl_file_path>
```
This script will display an interactive simulation environment window, where you can use the keyboard to control the robotic arm:

<table align="center">
  <tbody align="center">
  <tr>
    <th>Keys</th>
    <th colspan='2'>Command</th>
  </tr>
  <tr>
    <td><code>Q</code></td>
    <td colspan='2'>Reset environment</td>
  </tr>
  <tr>
    <td><code>Spacebar</code></td>
    <td>Toggle gripper (open/close)</td>
    <td><img src="image/spacebar.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>Up</code> / <code>Down</code> / <code>Left</code> / <code>Right</code></td>
    <td>Move horizontally in x-y plane</td>
    <td><img src="image/horizontal.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>.</code> / <code>;</code></td>
    <td>Move vertically</td>
    <td><img src="image/vertical.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>O</code> / <code>P</code></td>
    <td>Rotate (yaw)</td>
    <td><img src="image/o-p.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>Y</code> / <code>H</code></td>
    <td>Rotate (pitch)</td>
    <td><img src="image/y-h.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>E</code> / <code>R</code></td>
    <td>Rotate (roll)</td>
    <td><img src="image/e-r.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>[</code> / <code>]</code></td>
    <td colspan='2'>Switch to the previous/next view</td>
    
  </tr>
  <tr>
    <td><code>B</code></td>
    <td colspan='2'>Toggle arm/base mode (if applicable)</td>
    
  </tr>
  <tr>
    <td><code>S</code></td>
    <td colspan='2'>Switch active arm (if multi-armed robot)</td>
    
  </tr>
  <tr>
    <td><code>=</code></td>
    <td colspan='2'>Switch active robot (if multi-robot environment)</td>
    
  </tr>
  </tbody>
</table>

You need to manipulate the robotic arm to complete the task specified in the `language instruction` of the BDDL file, while avoiding any `cost` defined in the BDDL file.
<p align="center"><img src="image/data_collection_1.gif" width="300" height="300"/></p>

The collected demonstration data will be saved in `demonstration_data/` .

### Notes
- The collected data must form a continuous trajectory. Please minimize pauses during the collection process.
- If a mistake occurs during the process of operation, causing the trajectory to be interrupted (e.g., the object drops), please reset the environment and collect again.

## 2. Convert Data Format

The collected demonstration data only includes data of trajectory and scene, and does not contain images during the task execution process.
You can use `scripts/group_create_dataset.py` to convert the format of the demonstration data:
```bash
python scripts/group_create_dataset.py \
      --input-dir <directory_containing_demonstration_HDF5_files> \
      --output-dir <path_to_save_generated_dataset_files>
```
By replaying the original trajectory multiple times, the corresponding images will be stored in the dataset for model training. The conversion process takes a relatively long time.

### Notes
- During the conversion process, the simulation environment will be created based on the corresponding BDDL file, and the trajectory recorded in the demonstration data will be replay. Therefore, please ensure that the BDDL file and its assets remain unmodified when converting the data format; otherwise, it may cause program errors.
- The generated dataset files are large in size. Please ensure that there is sufficient space in the dataset storage location.

## 3. Regenerate Dataset

There are many empty actions (noops) in the trajectories of the original dataset, which can cause pauses in the trajectories and seriously affect the model's learning. We need to filter out these noops to ensure the continuity of the trajectories.
You can use `scripts/regenerate_dataset.py` to regenerate the dataset:
```bash
python scripts/regenerate_dataset.py \
        --task_suite <your_task_suite_name> \
        --raw_data_dir <path_to_raw_hdf5_dataset_files> \
        --target_dir <path_to_save_regenerated_dataset_files> \
        --task_levels level_of_tasks_to_regenerate
```
We first filter out trajectories with != 2 gripper transitions (multiple grasps are considered invalid). Then we filter out all noops and progressively add them in 4-step increments (4, 8, 12, 16) until the trajectory can successfully complete the task, ensuring that the trajectory contains only the minimum necessary noops.

### Notes
- The image observations will be saved at 256 * 256 resolution.
- The regenerated dataset only contains successful trajectories.

## 4. Convert Dataset to RLDS Format

Convert the regenerated dataset to RLDS (Reinforcement Learning Data Schema) format for integration with X-embodiment experimental frameworks. RLDS is a standard format developed by Google for storing robotics learning data.

### 4.1 Environment Setup

First, create a conda environment and install necessary dependencies:

```bash
cd rlds_dataset_builder
conda env create -f environment_ubuntu.yml
conda activate rlds_env

pip install -e .
```

Main dependencies include: `tensorflow`, `tensorflow_datasets`, `tensorflow_hub`, `apache_beam`, `matplotlib`, `plotly`, `wandb`, `h5py`, etc.

### 4.2 Configure Dataset Paths

Modify the `_split_paths()` method in `VLA_Arena/VLA_Arena_dataset_builder.py` to point to your regenerated dataset:

```python
def _split_paths(self):
    """Define filepaths for data splits."""
    return {
        "train": glob.glob("/path/to/your/regenerated/dataset/*.hdf5"),
    }
```

### 4.3 Dataset Feature Configuration

The dataset builder is already configured with the following features:

- **Observation Data**:
  - `image`: Main camera RGB image (256×256×3)
  - `wrist_image`: Wrist camera RGB image (256×256×3)  
  - `state`: Robot end-effector state (8D: 6D pose + 2D gripper state)
  - `joint_state`: Robot joint angles (7D)

- **Action Data**:
  - `action`: Robot end-effector action (7D)

- **Trajectory Information**:
  - `discount`: Discount factor (default 1.0)
  - `reward`: Reward signal (1.0 for final step in demonstrations)
  - `is_first`: Whether it's the first step of trajectory
  - `is_last`: Whether it's the last step of trajectory
  - `is_terminal`: Whether it's a terminal step
  - `language_instruction`: Language instruction

- **Metadata**:
  - `file_path`: Original data file path

### 4.4 Execute Conversion

Run the following command in the dataset directory to perform conversion:

```bash
cd VLA_Arena
tfds build --overwrite --data_dir ~/tensorflow_datasets
```

The conversion process will:
1. Read all HDF5 files
2. Parse each demonstration trajectory
3. Extract language instructions from filenames
4. Convert data to RLDS standard format
5. Save as TensorFlow Records format

### 4.5 Parallel Processing (Optional)

For large datasets, you can use multi-threaded parallel processing to improve conversion speed. Modify parameters in `VLA_Arena_dataset_builder.py`:

```python
N_WORKERS = 10              # Number of parallel worker threads
MAX_PATHS_IN_MEMORY = 10    # Number of files processed simultaneously in memory
```

### 4.6 Output Format

After conversion, the dataset will be saved in `~/tensorflow_datasets/VLA_Arena/` directory, containing:
- TFRecord files: Actual training data
- Metadata files: Dataset information and statistics
- Version information: Dataset version and release notes

### Notes

- Ensure sufficient disk space for storing the converted dataset
- The conversion process may take a long time, recommend running in background
- If memory issues occur during conversion, reduce the `MAX_PATHS_IN_MEMORY` parameter

## 5. Convert RLDS Dataset to LeRobot Format

Convert RLDS format dataset to LeRobot format for integration with LeRobot training framework. LeRobot is a robotics learning framework developed by Hugging Face, supporting multiple robot datasets and model training.

### 5.1 Environment Setup

First, create a new conda environment and install LeRobot-related dependencies:

```bash
# Create new conda environment
conda create -n lerobot_conversion python=3.9
conda activate lerobot_conversion

# Install dependencies
pip install -r conversion_requirements.txt
```

Main dependencies include:
- `tensorflow-datasets`: For reading RLDS format data
- `tensorflow`: TensorFlow core library
- `lerobot`: LeRobot framework (installed from GitHub)

### 5.2 Configure Conversion Parameters

Modify configuration variables in `scripts/convert.sh`:

```bash
# Set RLDS dataset path
DATA_DIR="/path/to/your/rlds/dataset"

# Set LeRobot output path, defaulting to "./lerobot_dataset"
HF_LEROBOT_HOME="/path/to/lerobot/datasets"

# Whether to push to Hugging Face Hub (optional)
PUSH_TO_HUB="false"
```

### 5.3 Dataset Feature Mapping

The conversion script will map RLDS data to LeRobot format:

- **Image Data**:
  - `image`: Main camera RGB image (256×256×3)
  - `wrist_image`: Wrist camera RGB image (256×256×3)

- **State Data**:
  - `state`: Robot end-effector state (8D: 6D pose + 2D gripper state)

- **Action Data**:
  - `actions`: Robot actions (7D)

- **Task Information**:
  - `task`: Language instruction (extracted from RLDS language_instruction)

### 5.4 Execute Conversion

Run the conversion script:

```bash
# Method 1: Use default configuration
./scripts/convert.sh

# Method 2: Specify data path
./scripts/convert.sh /path/to/your/rlds/dataset

# Method 3: Use environment variables
DATA_DIR=/path/to/your/rlds/dataset ./scripts/convert.sh
```

The conversion process will:
1. Validate input data path
2. Clean up old data in output directory
3. Create LeRobot dataset structure
4. Read RLDS trajectory data step by step
5. Convert to LeRobot format and save
6. Generate dataset metadata

### 5.5 Conversion Parameter Description

The following parameters can be adjusted in `convert_data_to_lerobot.py`:

```python
# Robot configuration
robot_type="panda"    # Robot type
fps=10               # Data sampling frequency

# Image processing configuration
image_writer_threads=10    # Number of image writing threads
image_writer_processes=5   # Number of image writing processes
```

### 5.6 Push to Hugging Face Hub (Optional)

If you need to push the dataset to Hugging Face Hub:

1. Modify `PUSH_TO_HUB="true"` in `convert.sh`
2. Ensure you're logged in to Hugging Face:
   ```bash
   huggingface-cli login
   ```
3. Run the conversion script

### 5.7 Output Format

After conversion, the dataset will be saved in the `HF_LEROBOT_HOME` directory

### Notes

- Ensure sufficient disk space for storing the converted dataset
- Image data will be compressed to save storage space
- The converted dataset can be directly used for LeRobot framework training
- If conversion fails, check if the RLDS dataset path is correct
