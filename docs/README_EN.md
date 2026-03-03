# VLA-Arena Documentation Table of Contents (English)

This document provides a comprehensive table of contents for all VLA-Arena documentation files.

## 📚 Complete Documentation Overview

### 1. Data Collection Guide
**File:** `data_collection.md`

A comprehensive guide for collecting demonstration data in custom scenes and converting data formats.

#### Table of Contents:
1. [Collect Demonstration Data](#1-collect-demonstration-data)
   - Interactive simulation environment setup
   - Keyboard controls for robotic arm manipulation
   - Data collection process and best practices
2. [Convert Data Format](#2-convert-data-format)
   - Converting demonstration data to training format
   - Image generation through trajectory replay
   - Dataset creation process
3. [Regenerate Dataset](#3-regenerate-dataset)
   - Filtering noop actions for trajectory continuity
   - Dataset optimization and validation
   - Quality assurance procedures
4. [Convert Dataset to RLDS Format](#4-convert-dataset-to-rlds-format)
   - RLDS format conversion
   - Dataset standardization
5. [Convert RLDS Dataset to LeRobot Format](#5-convert-rlds-dataset-to-lerobot-format)
   - LeRobot format conversion
   - Compatibility handling

---

### 2. Scene Construction Guide
**File:** `scene_construction.md`

Detailed guide for building custom task scenarios using BDDL (Behavior Domain Definition Language).

#### Table of Contents:
1. [BDDL File Structure](#1-bddl-file-structure)
   - Basic structure definition
   - Domain and problem definition
   - Language instruction specification
2. [Region Definition](#region-definition)
   - Spatial scope definition
   - Region parameters and configuration
3. [Object Definition](#object-definition)
   - Fixtures (static objects)
   - Manipulable objects
   - Objects of interest
   - Moving objects with motion types
4. [State Definition](#state-definition)
   - Initial state configuration
   - Goal state definition
   - Supported state predicates
5. [Image Effect Settings](#image-effect-settings)
   - Rendering effect configuration
   - Visual enhancement options
6. [Cost Constraints](#cost-constraints)
   - Penalty condition definition
   - Supported cost predicates
7. [Visualize BDDL File](#2-visualize-bddl-file)
   - Scene visualization process
   - Video generation workflow
8. [Assets](#3-assets)
   - Ready-made assets
   - Custom asset preparation
   - Asset registration process

---

### 3. Model Fine-tuning and Evaluation Guide
**File:** `finetuning_and_evaluation.md`

Comprehensive guide for fine-tuning and evaluating VLA models using VLA-Arena generated datasets. Supports OpenVLA, OpenVLA-OFT, Openpi, UniVLA, SmolVLA, and other models.

#### Table of Contents:
1. [General Models](#general-models)
   - Dependency installation
   - Model fine-tuning
   - Model evaluation
2. [Configuration File Notes](#configuration-file-notes)
   - Dataset path configuration
   - Model parameter settings
   - Training hyperparameter configuration

---

### 4. Task Asset Management Guide
**File:** `asset_management.md`

Comprehensive guide for packaging, sharing, and installing custom tasks and scenes.

#### Table of Contents:
1. [Overview](#1-overview)
   - Complete workflow: Design → Pack → Upload → Download → Install → Use
   - Key features and capabilities
   - What gets packaged
2. [Package a Single Task](#2-package-a-single-task)
   - Packaging commands and options
   - Automatic dependency detection
   - Examples and output
3. [Package a Task Suite](#3-package-a-task-suite)
   - Multi-task packaging
   - Suite organization
4. [Inspect a Package](#4-inspect-a-package)
   - Package content preview
   - Metadata inspection
5. [Install a Package](#5-install-a-package)
   - Installation procedures
   - Conflict handling
   - Options and flags
6. [Upload to Cloud](#6-upload-to-cloud)
   - HuggingFace Hub integration
   - Authentication setup
   - Automatic fallback methods
7. [Download from Cloud](#7-download-from-cloud)
   - Package discovery
   - Download and installation
8. [Uninstall a Package](#8-uninstall-a-package)
   - Safe removal procedures
9. [Package Structure](#9-package-structure)
   - `.vlap` file format
   - Manifest specification
10. [Troubleshooting](#10-troubleshooting)
    - Common issues and solutions
    - Best practices

---

## 🔧 CLI Entry Points

- `vla-arena`: unified training/evaluation CLI
- `vla-arena.download-tasks`: download task suites and assets from the Hub
- `vla-arena.manage-tasks`: pack/upload/download/install `.vlap` task packages

---

## 📁 Directory Structure

```
docs/
├── asset_management.md         # Task asset management guide (English)
├── asset_management_zh.md      # Task asset management guide (Chinese)
├── data_collection.md                    # Data collection guide (English)
├── data_collection_zh.md                 # Data collection guide (Chinese)
├── scene_construction.md                 # Scene construction guide (English)
├── scene_construction_zh.md              # Scene construction guide (Chinese)
├── finetuning_and_evaluation.md         # Model fine-tuning and evaluation guide (English)
├── finetuning_and_evaluation_zh.md      # Model fine-tuning and evaluation guide (Chinese)
├── README_EN.md                          # Documentation table of contents (English)
├── README_ZH.md                          # Documentation table of contents (Chinese)
└── image/                                # Documentation images and GIFs
```

---

## 🚀 Getting Started Workflow

### 1. Scene Construction
1. Read `scene_construction.md` for BDDL file structure
2. Define your task scenarios using BDDL syntax
3. Use `scripts/visualize_bddl.py` to preview scenes

### 2. Data Collection
1. Follow `data_collection.md` for demonstration collection
2. Use `scripts/collect_demonstration.py` for interactive data collection
3. Convert data format using `scripts/group_create_dataset.py`

### 3. Model Training
1. Follow `finetuning_and_evaluation.md` for the uv-only workflow
2. Fine-tune: `uv run --project envs/<model_name> vla-arena train --model <model_cli_name> --config vla_arena/configs/train/<model_cli_name>.yaml`
3. Evaluate: `uv run --project envs/<model_name> vla-arena eval --model <model_cli_name> --config vla_arena/configs/evaluation/<model_cli_name>.yaml`

> Note: the first `uv run` may take a while—it will create the environment and install dependencies automatically.

### 4. Task Sharing (Optional)
1. Follow `asset_management.md` to package your custom tasks
2. Use `vla-arena.manage-tasks` to upload/download/install packages
3. Share your task packages with the community
