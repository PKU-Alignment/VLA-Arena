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

import argparse

from .eval import eval_main
from .train import train_main


def main():
    parser = argparse.ArgumentParser('vla-arena CLI')
    sub = parser.add_subparsers(dest='cmd')

    # train
    train_p = sub.add_parser('train')
    train_p.add_argument('--model', required=True)
    train_p.add_argument('--config', default=None)
    train_p.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing checkpoint directory',
    )

    # eval
    eval_p = sub.add_parser('eval')
    eval_p.add_argument('--model', required=True)
    eval_p.add_argument('--config', default=None)

    args = parser.parse_args()

    if args.cmd == 'train':
        train_main(args)
    elif args.cmd == 'eval':
        eval_main(args)
    else:
        parser.print_help()
