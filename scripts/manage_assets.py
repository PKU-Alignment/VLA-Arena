#!/usr/bin/env python3
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
# ==============================================================================

"""
VLA-Arena Task Asset Manager - Command Line Tool

This script provides a convenient CLI for managing task assets in VLA-Arena.

Usage:
    python manage_assets.py <command> [options]

Commands:
    pack        - Pack a single task into a distributable package
    pack-suite  - Pack an entire task suite
    install     - Install a task package
    inspect     - Inspect package contents without installing
    upload      - Upload a package to HuggingFace Hub
    download    - Download a package from HuggingFace Hub
    list        - List available packages in the cloud
    uninstall   - Uninstall a task package

Examples:
    # Pack a single task
    python manage_assets.py pack path/to/task.bddl -o ./packages --author "Your Name"

    # Pack an entire task suite
    python manage_assets.py pack-suite robustness_dynamic_distractors -o ./packages

    # Inspect a package
    python manage_assets.py inspect my_task.vlap

    # Install a package
    python manage_assets.py install my_task.vlap

    # Upload to your cloud repo (required)
    python manage_assets.py upload my_task.vlap --repo username/task-assets

    # Download and install from a cloud repo
    python manage_assets.py download my_task --repo username/task-assets --install

    # List available packages in a repo
    python manage_assets.py list --repo username/task-assets
"""

import os
import sys


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_arena.vla_arena.utils.asset_manager import main


if __name__ == '__main__':
    main()
