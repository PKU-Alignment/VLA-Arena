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

"""
VLA-Arena Task Suite Downloader

Download and install task suites from HuggingFace Hub

Usage:
    # List available tasks
    python scripts/download_tasks.py list --repo username/vla-arena-tasks

    # Download a single task suite
    python scripts/download_tasks.py install robustness_dynamic_distractors --repo username/vla-arena-tasks

    # Download all task suites
    python scripts/download_tasks.py install-all --repo username/vla-arena-tasks
"""

import argparse
import os
import sys


# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vla_arena.vla_arena.utils.asset_manager import (
    TaskCloudManager,
    TaskInstaller,
)


try:
    from termcolor import colored
except ImportError:

    def colored(text, color=None, **kwargs):
        return text


# Default official repository
DEFAULT_REPO = 'vla-arena/tasks'


def list_available_tasks(repo_id: str = DEFAULT_REPO):
    """List available task suites"""
    print(f'\nQuerying repository: {repo_id}\n')

    try:
        cloud = TaskCloudManager(repo_id=repo_id)
        packages = cloud.list_packages()

        if packages:
            print(colored(f'✓ Found {len(packages)} task suites:\n', 'green'))
            for i, pkg in enumerate(packages, 1):
                print(f'  {i:2d}. {pkg}')
            print()
        else:
            print(colored('No task suites found', 'yellow'))
    except Exception as e:
        print(colored(f'❌ Error: {e}', 'red'))
        return []

    return packages


def install_task(
    task_name: str,
    repo_id: str = DEFAULT_REPO,
    token: str = None,
    overwrite: bool = False,
    skip_existing_assets: bool = False,
):
    """Download and install a single task suite"""
    print(f'\nDownloading task suite: {task_name}')
    print('=' * 80)

    try:
        cloud = TaskCloudManager(repo_id=repo_id)
        installer = TaskInstaller()

        # Download and install
        success = cloud.download_and_install(
            package_name=task_name,
            overwrite=overwrite,
            skip_existing_assets=skip_existing_assets,
            token=token,
        )

        if success:
            print(
                colored(
                    f'\n✓ Task suite {task_name} installed successfully!',
                    'green',
                ),
            )
        else:
            print(
                colored(
                    f'\n❌ Failed to install task suite {task_name}',
                    'red',
                ),
            )

        return success

    except Exception as e:
        print(colored(f'\n❌ Error: {e}', 'red'))
        return False


def install_all_tasks(
    repo_id: str = DEFAULT_REPO,
    token: str = None,
    overwrite: bool = False,
):
    """Download and install all task suites"""
    print('\nDownloading all task suites')
    print('=' * 80)

    # Get task list
    packages = list_available_tasks(repo_id)

    if not packages:
        print(colored('\nNo task suites available', 'yellow'))
        return

    print(f'\nPreparing to install {len(packages)} task suites')
    print(
        'Note: Shared assets will be automatically skipped if already installed.\n',
    )

    # Confirmation
    response = input('Continue? [y/N]: ')
    if response.lower() not in ['y', 'yes']:
        print('Cancelled')
        return

    # Install each task with skip_existing_assets=True to avoid conflicts
    successful = []
    failed = []

    for i, task_name in enumerate(packages, 1):
        print(f'\n[{i}/{len(packages)}] Installing: {task_name}')
        print('-' * 80)

        success = install_task(
            task_name=task_name,
            repo_id=repo_id,
            token=token,
            overwrite=overwrite,
            skip_existing_assets=True,  # Auto-skip shared assets
        )

        if success:
            successful.append(task_name)
        else:
            failed.append(task_name)

    # Display statistics
    print('\n' + '=' * 80)
    print(f'\n✓ Installation complete: {len(successful)}/{len(packages)}')

    if successful:
        print('\nSuccessfully installed:')
        for task in successful:
            print(f'  ✓ {task}')

    if failed:
        print('\nFailed to install:')
        for task in failed:
            print(f'  ✗ {task}')


def get_installed_tasks():
    """Get list of installed task suites"""
    from vla_arena.vla_arena import get_vla_arena_path

    bddl_root = get_vla_arena_path('bddl_files')

    installed = []
    if os.path.exists(bddl_root):
        for item in os.listdir(bddl_root):
            item_path = os.path.join(bddl_root, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it has BDDL files
                has_bddl = False
                for root, dirs, files in os.walk(item_path):
                    if any(f.endswith('.bddl') for f in files):
                        has_bddl = True
                        break
                if has_bddl:
                    installed.append(item)

    return sorted(installed)


def show_installed_tasks():
    """Display installed task suites"""
    installed = get_installed_tasks()

    if installed:
        print(
            colored(f'\n✓ {len(installed)} task suites installed:\n', 'green'),
        )
        for i, task in enumerate(installed, 1):
            print(f'  {i:2d}. {task}')
        print()
    else:
        print(colored('\nNo task suites installed', 'yellow'))
        print('\nUse the following command to install tasks:')
        print(
            f'  python scripts/download_tasks.py install-all --repo {DEFAULT_REPO}\n',
        )


def main():
    parser = argparse.ArgumentParser(
        description='VLA-Arena Task Suite Downloader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View installed tasks
  python scripts/download_tasks.py installed

  # List available tasks
  python scripts/download_tasks.py list --repo vla-arena/tasks

  # Install a single task
  python scripts/download_tasks.py install robustness_dynamic_distractors --repo vla-arena/tasks

  # Install multiple tasks
  python scripts/download_tasks.py install hazard_avoidance object_state_preservation --repo vla-arena/tasks

  # Install all tasks
  python scripts/download_tasks.py install-all --repo vla-arena/tasks
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # list command
    list_parser = subparsers.add_parser(
        'list',
        help='List available task suites',
    )
    list_parser.add_argument(
        '--repo',
        default=DEFAULT_REPO,
        help=f'HuggingFace repository ID (default: {DEFAULT_REPO})',
    )

    # installed command
    subparsers.add_parser('installed', help='Show installed task suites')

    # install command
    install_parser = subparsers.add_parser(
        'install',
        help='Install one or more task suites',
    )
    install_parser.add_argument(
        'task_names',
        nargs='+',
        help='Task suite name(s)',
    )
    install_parser.add_argument(
        '--repo',
        default=DEFAULT_REPO,
        help=f'HuggingFace repository ID (default: {DEFAULT_REPO})',
    )
    install_parser.add_argument('--token', help='HuggingFace API token')
    install_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files',
    )
    install_parser.add_argument(
        '--skip-existing-assets',
        action='store_true',
        help='Skip assets that already exist (useful when installing multiple suites)',
    )

    # install-all command
    install_all_parser = subparsers.add_parser(
        'install-all',
        help='Install all task suites',
    )
    install_all_parser.add_argument(
        '--repo',
        default=DEFAULT_REPO,
        help=f'HuggingFace repository ID (default: {DEFAULT_REPO})',
    )
    install_all_parser.add_argument('--token', help='HuggingFace API token')
    install_all_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files',
    )

    args = parser.parse_args()

    if args.command == 'list':
        list_available_tasks(repo_id=args.repo)

    elif args.command == 'installed':
        show_installed_tasks()

    elif args.command == 'install':
        task_names = args.task_names
        total = len(task_names)

        if total > 1:
            print(
                f'\nPreparing to install {total} task suites: {", ".join(task_names)}',
            )
            print(
                'Note: Shared assets will be automatically skipped if already installed.\n',
            )

        successful = []
        failed = []

        for i, task_name in enumerate(task_names, 1):
            if total > 1:
                print(f'\n[{i}/{total}] Installing: {task_name}')
                print('-' * 80)

            success = install_task(
                task_name=task_name,
                repo_id=args.repo,
                token=args.token,
                overwrite=args.overwrite,
                skip_existing_assets=getattr(
                    args,
                    'skip_existing_assets',
                    False,
                ),
            )

            if success:
                successful.append(task_name)
            else:
                failed.append(task_name)

        # Display statistics if multiple tasks
        if total > 1:
            print('\n' + '=' * 80)
            print(f'\n✓ Installation complete: {len(successful)}/{total}')

            if successful:
                print('\nSuccessfully installed:')
                for task in successful:
                    print(f'  ✓ {task}')

            if failed:
                print('\nFailed to install:')
                for task in failed:
                    print(f'  ✗ {task}')

    elif args.command == 'install-all':
        install_all_tasks(
            repo_id=args.repo,
            token=args.token,
            overwrite=args.overwrite,
        )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
