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

from vla_arena.config_paths import resolve_config_path


def eval_main(args):
    model = args.model
    try:
        # Support omitted --config and packaged vla_arena/configs/... refs.
        config_path = resolve_config_path(
            mode='eval', model=model, config_path=args.config
        )
    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(
            f'Unable to resolve eval config for model {model!r}: {e}\n'
            'If you installed from PyPI, initialize local uv projects first:\n'
            '  vla-arena.init-workspace --force',
        ) from e

    # 1. Dynamically get the physical path of the corresponding model evaluator.py file
    try:
        module_name = f'vla_arena.models.{model}.evaluator'
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None or module_spec.origin is None:
            raise ImportError(f'Cannot find module {module_name}')

    except ImportError as e:
        raise RuntimeError(
            f"Model '{model}' is not installed or evaluator script not found.\n"
            f'Try: pip install vla-arena[{model}]',
        ) from e

    # 2. Directly import the module and execute main
    module = importlib.import_module(module_name)
    # Pass config path string here, evaluator.py's main function will handle it
    module.main(cfg=config_path)
