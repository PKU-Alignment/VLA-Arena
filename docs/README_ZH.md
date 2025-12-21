# VLA-Arena 文档目录（中文版）

本文档提供了 VLA-Arena 所有文档文件的完整目录。

## 📚 完整文档概览

### 1. 数据收集指南
**文件：** `data_collection_zh.md`

在自定义场景中收集演示数据并转换数据格式的综合指南。

#### 目录结构：
1. [收集演示数据](#1-收集演示数据)
   - 交互式仿真环境设置
   - 机械臂操控的键盘控制
   - 数据收集过程和最佳实践
2. [转换数据格式](#2-转换数据格式)
   - 将演示数据转换为训练格式
   - 通过轨迹回放生成图像
   - 数据集创建过程
3. [重构数据集](#3-重构数据集)
   - 过滤空动作以确保轨迹连续性
   - 数据集优化和验证
   - 质量保证程序
4. [将数据集转换为rlds格式](#4-将数据集转换为rlds格式)
   - RLDS 格式转换
   - 数据集标准化
5. [将rlds数据集转换为lerobot格式](#5-将rlds数据集转换为lerobot格式)
   - LeRobot 格式转换
   - 兼容性处理

---

### 2. 场景构建指南
**文件：** `scene_construction_zh.md`

使用 BDDL（行为域定义语言）构建自定义任务场景的详细指南。

#### 目录结构：
1. [BDDL 文件结构](#1-bddl-文件结构)
   - 基本结构定义
   - 领域和问题定义
   - 语言指令规范
2. [区域定义](#区域定义)
   - 空间范围定义
   - 区域参数和配置
3. [对象定义](#对象定义)
   - 固定对象（静态对象）
   - 可操作对象
   - 关注对象
   - 具有运动类型的移动对象
4. [状态定义](#状态定义)
   - 初始状态配置
   - 目标状态定义
   - 支持的状态谓词
5. [图像效果设置](#图像效果设置)
   - 渲染效果配置
   - 视觉增强选项
6. [成本约束](#成本约束)
   - 惩罚条件定义
   - 支持的成本谓词
7. [可视化 BDDL 文件](#2-可视化-bddl-文件)
   - 场景可视化过程
   - 视频生成工作流
8. [资产](#3-资产)
   - 现成资产
   - 自定义资产准备
   - 资产注册过程

---

### 3. 模型微调与评估指南
**文件：** `finetuning_and_evaluation_zh.md`

使用 VLA-Arena 生成的数据集微调和评估 VLA 模型的综合指南。支持 OpenVLA、OpenVLA-OFT、Openpi、UniVLA、SmolVLA 等模型。

#### 目录结构：
1. [通用模型（OpenVLA、OpenVLA-OFT、UniVLA、SmolVLA）](#通用模型)
   - 依赖安装
   - 模型微调
   - 模型评估
2. [Openpi 模型](#openpi)
   - 环境配置（使用 uv）
   - 训练配置和运行
   - 策略服务器启动
   - 模型评估
3. [配置文件说明](#配置文件说明)
   - 数据集路径配置
   - 模型参数设置
   - 训练超参数配置

---

## 📁 目录结构

```
docs/
├── finetuning_and_evaluation.md         # 模型微调与评估指南（英文）
├── finetuning_and_evaluation_zh.md      # 模型微调与评估指南（中文）
├── data_collection.md          # 数据收集指南（英文）
├── data_collection_zh.md       # 数据收集指南（中文）
├── scene_construction.md       # 场景构建指南（英文）
├── scene_construction_zh.md    # 场景构建指南（中文）
├── asset_management.md         # 任务资产管理指南（英文）
├── asset_management_zh.md      # 任务资产管理指南（中文）
└── image/                      # 文档图片和 GIF
```

---

## 🚀 快速开始工作流

### 1. 场景构建
1. 阅读 `scene_construction_zh.md` 了解 BDDL 文件结构
2. 使用 BDDL 语法定义你的任务场景
3. 使用 `scripts/visualize_bddl.py` 预览场景

### 2. 数据收集
1. 按照 `data_collection_zh.md` 进行演示收集
2. 使用 `scripts/collect_demonstration.py` 进行交互式数据收集
3. 使用 `scripts/group_create_dataset.py` 转换数据格式

### 3. 模型训练与评估
1. 按照 `finetuning_and_evaluation_zh.md` 安装模型依赖
2. 使用 `vla-arena train` 命令进行模型微调
3. 根据您的需求配置训练参数
4. 使用 `vla-arena eval` 命令评估模型性能
5. 通过 WandB 监控训练进度
6. 分析结果并迭代改进模型

### 4. 模型评估
1. 按照 `evaluation_zh.md` 进行模型评估程序
2. 使用 `scripts/evaluate_policy.py` 进行全面评估
3. 分析结果并迭代改进模型

### 5. 任务分享（可选）
1. 按照 `asset_management_zh.md` 打包你的自定义任务
2. 使用 `scripts/manage_assets.py` 上传到云端
3. 与社区分享你的任务套件


