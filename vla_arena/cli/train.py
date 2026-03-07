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

import importlib
import importlib.util
import os
import subprocess
import sys

import torch


def train_main(args):
    model = args.model
    # Ensure config is an absolute path for easy reading by subprocesses
    config_path = os.path.abspath(str(args.config))

    # 1. Dynamically get the physical path of the corresponding model trainer.py file
    try:
        module_name = f'vla_arena.models.{model}.trainer'
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None or module_spec.origin is None:
            raise ImportError(f'Cannot find module {module_name}')

        script_path = module_spec.origin

    except ImportError as e:
        raise RuntimeError(
            f"Model '{model}' is not installed or trainer script not found.\n"
            f'Try: pip install vla-arena[{model}]',
        ) from e

    # 2. Special handling: openpi uses JAX, doesn't need torchrun
    if model == 'openpi':
        # === openpi uses JAX distributed training, directly call trainer ===
        print(f'[Launcher] Preparing JAX training for model: {model}')
        print(
            '[Launcher] JAX will automatically detect and use available GPUs'
        )

        # Collect override parameters
        override_kwargs = {}
        if hasattr(args, 'overwrite') and args.overwrite:
            override_kwargs['overwrite'] = True

        # Directly import the module and execute main
        module = importlib.import_module(module_name)
        # Pass config path string and override parameters here, trainer.py's main function will handle them
        module.main(config=config_path, **override_kwargs)
        return

    # 3. Check if currently launched by torchrun (check LOCAL_RANK environment variable)
    is_distributed = os.environ.get('LOCAL_RANK') is not None

    if is_distributed or model == 'smolvla':
        # === Case A: Already a Worker process (launched by torchrun) ===
        # Directly import the module and execute main
        module = importlib.import_module(module_name)
        # Pass config path string here, trainer.py's main function will handle it
        module.main(config=config_path)

    else:
        # === Case B: Main launch process (user runs vla-arena train ...) ===
        print(f'[Launcher] Preparing distributed training for model: {model}')

        # Get GPU count (support nproc specified in args, otherwise default to all visible GPUs)
        nproc_per_node = getattr(args, 'nproc', torch.cuda.device_count())
        nnodes = getattr(args, 'nnodes', 1)
        node_rank = getattr(args, 'node_rank', 0)
        master_addr = getattr(args, 'master_addr', '127.0.0.1')
        master_port = getattr(args, 'master_port', '29500')

        print(f'[Launcher] Launching torchrun with {nproc_per_node} GPUs...')

        # Build torchrun command
        cmd = [
            'torchrun',
            f'--nnodes={nnodes}',
            f'--nproc_per_node={nproc_per_node}',
            f'--node_rank={node_rank}',
            f'--master_addr={master_addr}',
            f'--master_port={master_port}',
            script_path,  # Target script: models/openvla/trainer.py
            f'--config={config_path}',  # Pass parameter: --config /path/to/yaml
        ]

        print(f"[Launcher] Executing: {' '.join(cmd)}")

        # Use subprocess to call torchrun
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f'[Launcher] Training failed with error code {e.returncode}')
            sys.exit(e.returncode)
