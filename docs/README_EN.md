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
1. [General Models (OpenVLA, OpenVLA-OFT, UniVLA, SmolVLA)](#general-models)
   - Dependency installation
   - Model fine-tuning
   - Model evaluation
2. [Openpi Model](#openpi)
   - Environment setup (using uv)
   - Training configuration and execution
   - Policy server startup
   - Model evaluation
3. [Configuration File Notes](#configuration-file-notes)
   - Dataset path configuration
   - Model parameter settings
   - Training hyperparameter configuration

---

### 4. Model Evaluation Guide
**File:** `evaluation.md`

Complete guide for evaluating VLA models and adding custom models to VLA-Arena.

#### Table of Contents:
1. [Quick Start](#quick-start)
   - Environment preparation
   - Basic evaluation commands
2. [Model Evaluation](#model-evaluation)
   - Supported models
   - Evaluation procedures
   - Performance metrics
   - Result interpretation
3. [Adding Custom Models](#adding-custom-models)
   - Custom model integration
   - Configuration requirements
   - Implementation guidelines
4. [Configuration Instructions](#configuration-instructions)
   - Detailed configuration options
   - Parameter descriptions
   - Best practices
5. [Troubleshooting](#troubleshooting)
   - Common issues and solutions
   - Debugging techniques
   - Performance optimization

---

### 5. Task Asset Management Guide
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

## 🔧 Script Files

### Fine-tuning Scripts
- **`finetune_openvla.sh`**: Standard OpenVLA fine-tuning script
- **`finetune_openvla_oft.sh`**: OpenVLA OFT fine-tuning script with advanced options

### Key Features:
- Automated dataset configuration
- Parameter validation
- Multi-GPU support
- Comprehensive error handling
- Flexible training options

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
1. Use `finetune_openvla.sh` or `finetune_openvla_oft.sh` for model fine-tuning
2. Configure training parameters according to your needs
3. Monitor training progress through WandB

### 4. Model Evaluation
1. Follow `evaluation.md` for model evaluation procedures
2. Use `scripts/evaluate_policy.py` for comprehensive evaluation
3. Analyze results and iterate on model improvements

### 5. Task Sharing (Optional)
1. Follow `asset_management.md` to package your custom tasks
2. Use `scripts/manage_assets.py` to upload to cloud
3. Share your task packages with the community
