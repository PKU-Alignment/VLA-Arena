#!/usr/bin/env python3
"""
Batch package all task suites into .vlap packages

Usage:
    python scripts/package_all_suites.py --output ./packages --author "VLA-Arena Team"
    
Optional arguments:
    --upload: Automatically upload to HuggingFace Hub after packaging
    --repo: HuggingFace repository ID (if using --upload)
    --token: HuggingFace API token (if using --upload)
"""

import argparse
import os
import sys


# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vla_arena.vla_arena.utils.asset_manager import (
    TaskCloudManager,
    TaskPackager,
    get_vla_arena_path,
)


def get_all_task_suites():
    """Get names of all task suites"""
    bddl_root = get_vla_arena_path('bddl_files')
    suites = []

    for item in os.listdir(bddl_root):
        item_path = os.path.join(bddl_root, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            suites.append(item)

    return sorted(suites)


def package_all_suites(
    output_dir: str,
    author: str = 'VLA-Arena Team',
    email: str = '',
    upload: bool = False,
    repo_id: str = None,
    token: str = None,
):
    """
    Package all task suites

    Args:
        output_dir: Output directory
        author: Author name
        email: Author email
        upload: Whether to upload to HuggingFace Hub
        repo_id: HuggingFace repository ID
        token: HuggingFace API token
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all task suites
    suites = get_all_task_suites()

    print(f'\nFound {len(suites)} task suites:\n')
    for i, suite in enumerate(suites, 1):
        print(f'  {i}. {suite}')

    print('\nStarting packaging...\n')
    print('=' * 80)

    packager = TaskPackager()
    packaged_files = []

    # Package each suite
    for i, suite_name in enumerate(suites, 1):
        print(f'\n[{i}/{len(suites)}] Packaging: {suite_name}')
        print('-' * 80)

        try:
            # Generate description
            description = f'{suite_name} task suite from VLA-Arena benchmark'

            # Package
            package_path = packager.pack_task_suite(
                task_suite_name=suite_name,
                output_dir=output_dir,
                author=author,
                email=email,
                description=description,
            )

            packaged_files.append((suite_name, package_path))

        except Exception as e:
            print(f'  ❌ Error: {e}')
            continue

    print('\n' + '=' * 80)
    print(f'\n✓ Packaging complete! {len(packaged_files)}/{len(suites)} suites')
    print(f'\nPackages saved to: {output_dir}\n')

    # Display packaging statistics
    total_size = 0
    for suite_name, package_path in packaged_files:
        if os.path.exists(package_path):
            size = os.path.getsize(package_path)
            total_size += size
            print(f'  - {suite_name}: {size / 1024 / 1024:.2f} MB')

    print(f'\nTotal size: {total_size / 1024 / 1024:.2f} MB')

    # Upload to HuggingFace Hub
    if upload:
        if not repo_id:
            print('\n⚠ Need to provide --repo parameter to upload to HuggingFace Hub')
            return packaged_files

        print(f'\nUploading to HuggingFace Hub: {repo_id}')
        print('=' * 80)

        cloud = TaskCloudManager(repo_id=repo_id)

        for i, (suite_name, package_path) in enumerate(packaged_files, 1):
            print(f'\n[{i}/{len(packaged_files)}] Uploading: {suite_name}')
            print('-' * 80)

            try:
                url = cloud.upload(
                    package_path=package_path,
                    token=token,
                    private=False,
                )
                print(f'  ✓ Upload successful: {url}')
            except Exception as e:
                print(f'  ❌ Upload failed: {e}')

        print('\n' + '=' * 80)
        print('\n✓ Upload complete!')

    return packaged_files


def main():
    parser = argparse.ArgumentParser(
        description='Batch package all VLA-Arena task suites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Package only
  python scripts/package_all_suites.py --output ./packages

  # Package and upload
  python scripts/package_all_suites.py --output ./packages \\
      --upload --repo username/vla-arena-tasks --token hf_xxx
        """,
    )

    parser.add_argument(
        '--output',
        '-o',
        default='./packages',
        help='Output directory (default: ./packages)',
    )
    parser.add_argument(
        '--author',
        default='VLA-Arena Team',
        help='Author name',
    )
    parser.add_argument(
        '--email',
        default='',
        help='Author email',
    )
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Automatically upload to HuggingFace Hub after packaging',
    )
    parser.add_argument(
        '--repo',
        help='HuggingFace repository ID (e.g., username/vla-arena-tasks)',
    )
    parser.add_argument(
        '--token',
        help='HuggingFace API token',
    )

    args = parser.parse_args()

    package_all_suites(
        output_dir=args.output,
        author=args.author,
        email=args.email,
        upload=args.upload,
        repo_id=args.repo,
        token=args.token,
    )


if __name__ == '__main__':
    main()
