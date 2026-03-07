# 任务资产管理指南

VLA-Arena 提供了一套完整的资产管理系统，用于打包、分享和安装自定义任务和场景。本指南将帮助你了解如何使用资产管理器与社区分享你的自定义任务。

## 目录
1. [概述](#1-概述)
2. [打包单个任务](#2-打包单个任务)
3. [打包任务套件](#3-打包任务套件)
4. [检查包内容](#4-检查包内容)
5. [安装包](#5-安装包)
6. [上传到云端](#6-上传到云端)
7. [从云端下载](#7-从云端下载)
8. [卸载包](#8-卸载包)
9. [包结构说明](#9-包结构说明)
10. [故障排除](#10-故障排除)

## 1. 概述

资产管理系统提供了分享自定义任务的完整工作流：

```
设计 → 打包 → 上传 → 下载 → 安装 → 使用
```

### 核心功能
- **自动依赖检测**：自动查找所有需要的资产（物体、纹理、网格、Problem 类、场景 XML）
- **自包含包**：所有依赖项打包到单个 `.vlap` 文件中
- **云端集成**：上传到 HuggingFace Hub 或你自己的仓库
- **版本控制**：带校验和的清单确保完整性
- **冲突检测**：安装前警告已存在的文件

### 打包内容
- **BDDL 文件**：包含物体、固定装置和目标的任务定义
- **初始化文件**：初始状态文件（`.pruned_init`）
- **物体资产**：3D 模型、纹理、碰撞网格
- **Problem 类**：自定义 Python 环境定义
- **场景 XML**：包含纹理和网格的 MuJoCo 场景描述
- **元数据**：作者信息、描述、校验和

## 2. 打包单个任务

要打包单个任务，你需要 BDDL 文件路径：

```bash
python scripts/manage_assets.py pack <bddl文件路径> \
    -o ./packages \
    --author "你的名字" \
    --email "your.email@example.com" \
    --description "一个自定义的拾取和放置任务"
```

### 示例
```bash
python scripts/manage_assets.py pack \
    vla_arena/vla_arena/bddl_files/robustness_static_distractors/level_0/pick_up_the_banana_and_put_it_on_the_plate.bddl \
    -o ./packages \
    --author "VLA-Arena Team" \
    --description "带静态干扰物的拾取香蕉任务"
```

### 选项
- `-o, --output`：输出目录（默认：当前目录）
- `--name`：自定义包名称（默认：从 BDDL 文件名派生）
- `--author`：作者名字
- `--email`：作者邮箱
- `--description`：任务描述
- `--init`：手动指定初始化文件（默认自动检测）
- `--no-assets`：跳过包含资产（用于测试）

## 3. 打包任务套件

要打包包含多个任务的整个任务套件：

```bash
python scripts/manage_assets.py pack-suite <套件名称> \
    -o ./packages \
    --author "你的名字" \
    --description "任务套件的描述"
```

### 示例
```bash
python scripts/manage_assets.py pack-suite robustness_static_distractors \
    -o ./packages \
    --author "VLA-Arena Team" \
    --description "静态干扰物鲁棒性测试"
```

## 4. 检查包内容

在安装之前，你可以检查包的内容：

```bash
python scripts/manage_assets.py inspect <包路径>
```

### 示例
```bash
python scripts/manage_assets.py inspect ./packages/pick_up_the_banana_and_put_it_on_the_plate.vlap
```

## 5. 安装包

要将包安装到你的 VLA-Arena 安装中：

```bash
python scripts/manage_assets.py install <包路径>
```

### 示例
```bash
python scripts/manage_assets.py install ./packages/pick_up_the_banana_and_put_it_on_the_plate.vlap
```

### 选项
- `--overwrite`：覆盖已存在的文件（默认：跳过）
- `--skip-assets`：仅安装 BDDL 和初始化文件，跳过资产
- `--dry-run`：显示将要安装的内容但不实际安装

### 冲突处理
如果文件已存在，安装程序会警告你：
```
⚠ Conflicts detected:
  - Asset: ceramic_plate (already exists)
  - Asset: banana (already exists)
  - Asset: toy_train (already exists)

Use --overwrite to replace existing files.
```

## 6. 上传到云端

将包上传到 HuggingFace Hub（或你自己的仓库）：

```bash
python scripts/manage_assets.py upload <包路径> \
    --repo 用户名/仓库名
```

### 前置要求
1. **HuggingFace 账户**：在 https://huggingface.co 注册
2. **创建仓库**：
   - 访问 https://huggingface.co/new-dataset
   - 创建数据集仓库（例如：`username/vla-arena-tasks`）
3. **获取访问令牌**：
   - 访问 https://huggingface.co/settings/tokens
   - 创建具有 **write** 权限的令牌
4. **安装 Git LFS**（支持备用方法）

### 示例
```bash
# 将令牌设置为环境变量（推荐）
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

# 上传包
python scripts/manage_assets.py upload \
    ./packages/pick_up_the_banana_and_put_it_on_the_plate.vlap \
    --repo username/vla-arena-tasks
```

### 使用令牌参数
```bash
python scripts/manage_assets.py upload \
    ./packages/pick_up_the_banana_and_put_it_on_the_plate.vlap \
    --repo username/vla-arena-tasks \
    --token hf_your_token_here \
    --private  # 将仓库设为私有（可选）
```

### 输出
```
Uploading via HuggingFace API...
✓ Uploaded: https://huggingface.co/datasets/username/vla-arena-tasks/blob/main/packages/pick_up_the_banana_and_put_it_on_the_plate.vlap
```

### 自动回退
如果 API 上传失败（例如，由于速率限制），系统会自动使用 Git LFS 重试：
```
⚠ API upload failed: 403 Forbidden
Retrying with Git LFS method...

Using Git LFS upload method...
  Cloning repository...
  Setting up Git LFS...
  Copying pick_up_the_banana_and_put_it_on_the_plate.vlap...
  Creating commit...
  Pushing to HuggingFace...
✓ Uploaded via Git LFS: https://huggingface.co/...
```

## 7. 从云端下载

从 HuggingFace Hub 下载并可选地安装包：

```bash
python scripts/manage_assets.py download <包名称> \
    --repo 用户名/仓库名 \
    --install
```

### 示例：仅下载
```bash
python scripts/manage_assets.py download pick_up_the_banana_and_put_it_on_the_plate \
    --repo username/vla-arena-tasks \
    -o ./downloaded_packages
```

### 示例：下载并安装
```bash
python scripts/manage_assets.py download pick_up_the_banana_and_put_it_on_the_plate \
    --repo username/vla-arena-tasks \
    --install \
    --overwrite
```

### 列出可用包
```bash
python scripts/manage_assets.py list --repo username/vla-arena-tasks
```

### 输出
```
Available packages:
  - pick_up_the_banana_and_put_it_on_the_plate
  - robustness_static_distractors
  - long_horizon
```

## 8. 卸载包

要移除已安装的包：

```bash
python scripts/manage_assets.py uninstall <包名称>
```

### 示例
```bash
python scripts/manage_assets.py uninstall pick_up_the_banana_and_put_it_on_the_plate
```

### 输出
```
Uninstalling: pick_up_the_banana_and_put_it_on_the_plate
  ✓ Removed BDDL files
  ✓ Removed init files
  ⚠ Assets not removed (shared with other tasks)

✓ Uninstalled: pick_up_the_banana_and_put_it_on_the_plate
```

### 选项
- `--remove-assets`：同时移除关联的资产文件（请谨慎使用，因为资产可能被共享）

## 9. 包结构说明

`.vlap` 包是一个 ZIP 文件，具有以下结构：

```
package_name.vlap
├── manifest.json              # 元数据和校验和
├── bddl_files/               # 任务定义
│   └── task_name.bddl
├── init_files/               # 初始状态
│   └── task_name.pruned_init
├── problems/                 # 自定义 Problem 类
│   └── tabletop_manipulation.py
└── assets/                   # 所有资产
    ├── stable_scanned_objects/
    │   └── banana/
    │       ├── banana.xml
    │       ├── visual/
    │       │   ├── model_normalized_0.obj
    │       │   └── image0.png
    │       └── collision/
    │           └── model_normalized_collision_22.obj
    ├── scenes/               # 场景 XML
    │   └── tabletop_warm_style.xml
    └── textures/             # 场景纹理
        └── martin_novak_wood_table.png
```

### 清单格式
```json
{
  "package_name": "pick_up_the_banana_and_put_it_on_the_plate",
  "version": "1.0.0",
  "task_name": "Tabletop_Manipulation",
  "description": "带静态干扰物的拾取香蕉任务",
  "author": "Alice Smith",
  "email": "alice@example.com",
  "created_at": "2024-12-15T10:30:00",
  "bddl_files": ["pick_up_the_banana_and_put_it_on_the_plate.bddl"],
  "init_files": ["pick_up_the_banana_and_put_it_on_the_plate.pruned_init"],
  "problem_files": ["tabletop_manipulation.py"],
  "scene_files": ["tabletop_warm_style.xml"],
  "assets": [
    {
      "object_type": "banana",
      "relative_path": "stable_scanned_objects/banana/banana.xml",
      "checksum": "a1b2c3d4..."
    }
  ],
  "objects": ["banana", "ceramic_plate", "toy_train"],
  "total_size_bytes": 7549747
}
```

## 10. 故障排除

### 问题："Unknown object type" 警告
**症状**：关于像 `table` 或 `floor` 等固定装置的警告
```
[Warning] Unknown object type: table
```
**解决方案**：这是正常的。像 `table`、`floor` 和 `main_table` 这样的固定装置是环境 arena 的一部分，不会单独打包。

### 问题：NumPy 兼容性错误
**症状**：`AttributeError: _ARRAY_API not found`
```bash
AttributeError: numpy.core.multiarray has no attribute _ARRAY_API
```
**解决方案**：降级 NumPy 到兼容版本：
```bash
pip install "numpy>=1.24,<2"
```

### 问题：HuggingFace 403 Forbidden
**症状**：上传失败，提示 "403 Forbidden: Your storage patterns tripped our internal systems"
```
403 Forbidden: Your storage patterns tripped our internal systems!
```
**解决方案**：
1. **等待并重试**：HuggingFace 有速率限制；等待 10-30 分钟
2. **联系支持**：发送邮件到 website@huggingface.co 说明你的使用情况
3. **安装 Git LFS**：系统会自动使用 Git LFS 重试
4. **使用个人仓库**：上传到你的个人账户而不是组织账户

### 问题：Git LFS 未找到
**症状**：`git: 'lfs' is not a git command`
**解决方案**：安装 Git LFS：
```bash
# macOS
brew install git-lfs
git lfs install

# Ubuntu/Debian
sudo apt-get install git-lfs
git lfs install

# Windows
# 从 https://git-lfs.github.com/ 下载
```

### 问题：包安装冲突
**症状**：文件已存在
```
⚠ Conflicts detected:
  - Asset: banana (already exists)
  - Asset: ceramic_plate (already exists)

Use --overwrite to replace existing files.
```
**解决方案**：
1. **跳过冲突**：默认行为，保留现有文件
2. **强制覆盖**：使用 `--overwrite` 标志
3. **先进行试运行**：使用 `--dry-run` 查看将要安装的内容

### 问题：缺少依赖
**症状**：安装后任务无法运行
**解决方案**：确保 robosuite 和所有依赖项已安装：
```bash
cd vla_arena
pip install -e .
```

### 问题：路径解析错误
**症状**：无法找到 BDDL 或资产文件
**解决方案**：VLA-Arena 现在使用相对路径。确保你从项目根目录运行命令：
```bash
cd /path/to/VLA-Arena
python scripts/manage_assets.py <命令>
```

## 注意事项和最佳实践

### 打包时
- **先测试**：打包前始终在本地测试你的任务
- **描述性名称**：使用清晰、描述性的包名称
- **作者信息**：包含作者和联系信息
- **文档**：添加解释任务功能的描述

### 上传时
- **令牌安全**：永远不要将令牌提交到版本控制
- **使用环境变量**：将令牌导出为 `HUGGING_FACE_HUB_TOKEN`
- **仓库命名**：使用描述性的仓库名称，如 `username/vla-arena-tasks`
- **私有 vs 公开**：考虑你的任务是否应该公开

### 安装时
- **先检查**：使用 `inspect` 命令在安装前检查
- **备份**：使用 `--overwrite` 前备份现有任务
- **试运行**：使用 `--dry-run` 预览更改
- **资产**：卸载时谨慎使用 `--remove-assets`

### 包大小考虑
- 单个任务：通常 5-10 MB
- 任务套件：可能 30-60 MB 或更多
- HuggingFace 免费层：50 GB 存储限制
- 考虑将大型套件拆分为较小的包

## 示例工作流

从创建自定义任务到分享的完整工作流：

```bash
# 1. 创建你的自定义任务（BDDL 文件、Problem 类、资产）
# ... （参见场景构建指南）

# 2. 打包你的任务
python scripts/manage_assets.py pack \
    vla_arena/vla_arena/bddl_files/my_custom_task/my_task.bddl \
    -o ./packages \
    --author "你的名字" \
    --email "you@example.com" \
    --description "我的很棒的自定义任务"

# 3. 检查包
python scripts/manage_assets.py inspect ./packages/my_task.vlap

# 4. 上传到 HuggingFace
export HUGGING_FACE_HUB_TOKEN="your_token"
python scripts/manage_assets.py upload \
    ./packages/my_task.vlap \
    --repo yourusername/vla-arena-custom-tasks

# 5. 与他人分享下载命令
echo "要使用我的任务，运行："
echo "python scripts/manage_assets.py download my_task --repo yourusername/vla-arena-custom-tasks --install"
```

其他人现在可以用一个命令安装你的任务：
```bash
python scripts/manage_assets.py download my_task \
    --repo yourusername/vla-arena-custom-tasks \
    --install
```

## 其他资源

- [场景构建指南](scene_construction_zh.md) - 如何创建自定义任务
- [数据收集指南](data_collection_zh.md) - 如何收集演示数据
- [评估指南](evaluation_zh.md) - 如何评估策略
- [HuggingFace Hub 文档](https://huggingface.co/docs/hub/index) - 云存储
