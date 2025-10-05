# 数据收集指南

VLA-Arena 提供了一套用于在自定义的场景中搜集数据并转换搜集到的数据格式的完整框架。本指南将帮助您了解如何在已有的场景中搜集数据，并将其转换为可用的数据集。

## 目录
1. [收集演示数据](#1-收集演示数据)
2. [转换数据格式](#2-转换数据格式)
3. [重构数据集](#3-重构数据集)
4. [将数据集转换为rlds格式](#4-将数据集转换为rlds格式)
5. [将rlds数据集转换为lerobot格式](#5-将rlds数据集转换为lerobot格式)

## 1. 收集演示数据
您可以使用 `scripts/collect_demonstration.py` 在仿真环境中收集演示数据：
```bash
python scripts/collect_demonstration.py --bddl-file <您的bddl文件路径>
```
这个脚本会显示一个可交互的仿真环境窗口，您可以使用键盘来操控机械臂。

<table align="center">
  <tbody align="center">
  <tr>
    <th>按键</th>
    <th colspan='2'>命令</th>
  </tr>
  <tr>
    <td><code>Q</code></td>
    <td colspan='2'>重置环境</td>
  </tr>
  <tr>
    <td><code>空格键</code></td>
    <td>切换夹爪（打开/关闭）</td>
    <td><img src="image/spacebar.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>上</code> / <code>下</code> / <code>左</code> / <code>右</code></td>
    <td>在 x-y 平面内水平移动</td>
    <td><img src="image/horizontal.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>.</code> / <code>;</code></td>
    <td>垂直移动</td>
    <td><img src="image/vertical.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>O</code> / <code>P</code></td>
    <td>旋转（偏转）</td>
    <td><img src="image/o-p.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>Y</code> / <code>H</code></td>
    <td>旋转（俯仰）</td>
    <td><img src="image/y-h.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>E</code> / <code>R</code></td>
    <td>旋转（翻滚）</td>
    <td><img src="image/e-r.gif" style="width:100px;height:auto;"></td>
  </tr>
  <tr>
    <td><code>[</code> / <code>]</code></td>
    <td colspan='2'>切换到上一个/下一个视图</td>
    
  </tr>
  <tr>
    <td><code>B</code></td>
    <td colspan='2'>切换手臂/基座模式（如适用）</td>
    
  </tr>
  <tr>
    <td><code>S</code></td>
    <td colspan='2'>切换活动手臂（如果是多臂机器人）</td>
    
  </tr>
  <tr>
    <td><code>=</code></td>
    <td colspan='2'>切换活动机器人（如果是多机器人环境）</td>
    
  </tr>
  </tbody>
</table>

您需要操作机械臂完成 BDDL 文件中 `language instruction` 给出的任务，同时避免产生任何 BDDL 文件中所定义的 `cost`。
<p align="center"><img src="image/data_collection_1.gif" width="300" height="300"/></p>

收集的演示数据将保存在 `demonstration_data/` 中。

### 注意事项
- 搜集的数据必须是一条连贯的轨迹，请尽可能减少搜集过程中的停顿。
- 如果在操作机械臂完成任务的过程中出现了失误导致轨迹中断（如物体掉落），请重置环境重新搜集。

## 2. 转换数据格式

收集的演示数据仅包括轨迹和场景的数据，不包含任务执行过程中的图像。
您可以使用 `scripts/group_create_dataset.py` 转换演示数据的格式：
```bash
python scripts/group_create_dataset.py \
      --input-dir <包含演示 HDF5 文件的目录> \
      --output-dir <保存生成的数据集文件的路径>
```
通过多次回放原始轨迹来将对应的图像存入数据集中，用于模型的训练。转换过程所需时间较长。

### 注意事项
- 转换过程中会根据对应的 BDDL 文件创建仿真环境，重新运行演示数据中记录的轨迹，所以请保证转换数据格式时 BDDL 文件与其中的资产都没有被修改，否则可能会导致程序报错；
- 生成的数据集文件体积较大，请保证数据集存储位置有足够的空间。

## 3. 重构数据集

原始数据集的轨迹中有许多空动作（noop），这会导致轨迹出现停顿，严重影响模型的学习。我们需要过滤掉这些空动作，以确保轨迹的连续性。
您可以使用 `scripts/regenerate_dataset.py` 重构数据集：
```bash
python scripts/regenerate_dataset.py \
        --task_suite <您的任务套件名称> \
        --raw_data_dir <原始 HDF5 数据集文件的路径> \
        --target_dir <保存重构的数据集文件的路径> \
        --task_levels 要重构的任务级别
```
重构过程中，我们首先会过滤掉夹爪开合次数不等于 2 的轨迹（多次抓取，无效），然后筛选出所有的空动作，逐步添加空动作，以 4 步为增量（4、8、12、16）添加，直至轨迹可以成功完成任务，保证轨迹中仅包含最少的必要空动作。

### 注意事项
- 数据集中图像的分辨率为 256 * 256；
- 重构后数据集中仅包含成功的轨迹。

## 4. 将数据集转换为rlds格式

将重构后的数据集转换为RLDS（Reinforcement Learning Data Schema）格式，以便与X-embodiment实验框架集成。RLDS是Google开发的标准格式，用于存储机器人学习数据。

### 4.1 环境准备

首先创建conda环境并安装必要的依赖：

```bash
cd rlds_dataset_builder
conda env create -f environment_ubuntu.yml
conda activate rlds_env

pip install -e .
```

主要依赖包包括：`tensorflow`、`tensorflow_datasets`、`tensorflow_hub`、`apache_beam`、`matplotlib`、`plotly`、`wandb`、`h5py`等。

### 4.2 配置数据集路径

修改 `VLA_Arena/VLA_Arena_dataset_builder.py` 文件中的 `_split_paths()` 方法，将数据路径指向您的重构数据集：

```python
def _split_paths(self):
    """Define filepaths for data splits."""
    return {
        "train": glob.glob("/path/to/your/regenerated/dataset/*.hdf5"),
    }
```

### 4.3 数据集特征配置

数据集构建器已经配置了以下特征：

- **观察数据**：
  - `image`: 主摄像头RGB图像 (256×256×3)
  - `wrist_image`: 手腕摄像头RGB图像 (256×256×3)  
  - `state`: 机器人末端执行器状态 (8维：6D位姿 + 2D夹爪状态)
  - `joint_state`: 机器人关节角度 (7维)

- **动作数据**：
  - `action`: 机器人末端执行器动作 (7维)

- **轨迹信息**：
  - `discount`: 折扣因子 (默认1.0)
  - `reward`: 奖励信号 (演示数据最后一步为1.0)
  - `is_first`: 是否为轨迹第一步
  - `is_last`: 是否为轨迹最后一步
  - `is_terminal`: 是否为终止步骤
  - `language_instruction`: 语言指令

- **元数据**：
  - `file_path`: 原始数据文件路径

### 4.4 执行转换

在数据集目录中运行以下命令进行转换：

```bash
cd VLA_Arena
tfds build --overwrite --data_dir ~/tensorflow_datasets
```

转换过程会：
1. 读取所有HDF5文件
2. 解析每个演示轨迹
3. 从文件名提取语言指令
4. 将数据转换为RLDS标准格式
5. 保存为TensorFlow Records格式

### 4.5 并行处理（可选）

对于大型数据集，可以使用多线程并行处理提高转换速度。修改 `VLA_Arena_dataset_builder.py` 中的参数：

```python
N_WORKERS = 10              # 并行工作线程数
MAX_PATHS_IN_MEMORY = 10    # 内存中同时处理的文件数
```

### 4.6 输出格式

转换完成后，数据集将保存在 `~/tensorflow_datasets/VLA_Arena/` 目录下，包含：
- TFRecord文件：实际的训练数据
- 元数据文件：数据集信息和统计
- 版本信息：数据集版本和发布说明

### 注意事项

- 确保有足够的磁盘空间存储转换后的数据集
- 转换过程可能需要较长时间，建议在后台运行
- 如果转换过程中出现内存不足，可以减少 `MAX_PATHS_IN_MEMORY` 参数

## 5. 将rlds数据集转换为lerobot格式

将RLDS格式的数据集转换为LeRobot格式，以便与LeRobot训练框架集成。LeRobot是Hugging Face开发的机器人学习框架，支持多种机器人数据集和模型训练。

### 5.1 环境准备

首先创建新的conda环境并安装LeRobot相关依赖：

```bash
# 创建新的conda环境
conda create -n lerobot_conversion python=3.9
conda activate lerobot_conversion

# 安装依赖包
pip install -r conversion_requirements.txt
```

主要依赖包括：
- `tensorflow-datasets`: 用于读取RLDS格式数据
- `tensorflow`: TensorFlow核心库
- `lerobot`: LeRobot框架（从GitHub安装）

### 5.2 配置转换参数

修改 `scripts/convert.sh` 脚本中的配置变量：

```bash
# 设置RLDS数据集路径
DATA_DIR="/path/to/your/rlds/dataset"

# 设置LeRobot输出路径，默认为 "./lerobot_dataset"
HF_LEROBOT_HOME="/path/to/lerobot/datasets"

# 是否推送到Hugging Face Hub（可选）
PUSH_TO_HUB="false"
```

### 5.3 数据集特征映射

转换脚本会将RLDS数据映射到LeRobot格式：

- **图像数据**：
  - `image`: 主摄像头RGB图像 (256×256×3)
  - `wrist_image`: 手腕摄像头RGB图像 (256×256×3)

- **状态数据**：
  - `state`: 机器人末端执行器状态 (8维：6D位姿 + 2D夹爪状态)

- **动作数据**：
  - `actions`: 机器人动作 (7维)

- **任务信息**：
  - `task`: 语言指令（从RLDS的language_instruction提取）

### 5.4 执行转换

运行转换脚本：

```bash
# 方法1：使用默认配置
./scripts/convert.sh

# 方法2：指定数据路径
./scripts/convert.sh /path/to/your/rlds/dataset

# 方法3：使用环境变量
DATA_DIR=/path/to/your/rlds/dataset ./scripts/convert.sh
```

转换过程会：
1. 验证输入数据路径
2. 清理输出目录中的旧数据
3. 创建LeRobot数据集结构
4. 逐条读取RLDS轨迹数据
5. 转换为LeRobot格式并保存
6. 生成数据集元数据

### 5.5 转换参数说明

在 `convert_data_to_lerobot.py` 中可以调整以下参数：

```python
# 机器人配置
robot_type="panda"    # 机器人类型
fps=10               # 数据采样频率

# 图像处理配置
image_writer_threads=10    # 图像写入线程数
image_writer_processes=5   # 图像写入进程数
```

### 5.6 推送到Hugging Face Hub（可选）

如果需要将数据集推送到Hugging Face Hub：

1. 修改 `convert.sh` 中的 `PUSH_TO_HUB="true"`
2. 确保已登录Hugging Face：
   ```bash
   huggingface-cli login
   ```
3. 运行转换脚本

### 5.7 输出格式

转换完成后，数据集将保存在 `HF_LEROBOT_HOME` 目录下

### 注意事项

- 确保有足够的磁盘空间存储转换后的数据集
- 图像数据会进行压缩以节省存储空间
- 转换后的数据集可以直接用于LeRobot框架的训练
- 如果转换失败，检查RLDS数据集路径是否正确

