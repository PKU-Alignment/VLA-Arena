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
import random
from pathlib import Path

import h5py
import numpy as np


def copy_hdf5_group(source_group, target_group):
    """
    Recursively copy all data and attributes from an HDF5 group.

    Args:
        source_group: Source HDF5 group
        target_group: Target HDF5 group
    """
    # Copy all attributes
    for key, value in source_group.attrs.items():
        target_group.attrs[key] = value

    # Copy all datasets and subgroups
    for key in source_group.keys():
        source_item = source_group[key]
        if isinstance(source_item, h5py.Dataset):
            # Copy dataset
            target_group.create_dataset(key, data=source_item[:])
        elif isinstance(source_item, h5py.Group):
            # Recursively copy subgroup
            target_subgroup = target_group.create_group(key)
            copy_hdf5_group(source_item, target_subgroup)


def sample_hdf5_file(input_file, output_file, sample_ratio, random_seed=None):
    """
    Randomly sample a certain proportion of demos from an HDF5 file and create a new HDF5 file.

    Args:
        input_file: Input HDF5 file path
        output_file: Output HDF5 file path
        sample_ratio: Sampling ratio (0.0 - 1.0)
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    print(f'Processing file: {input_file}')

    # Open input file
    try:
        with h5py.File(input_file, 'r') as f_in:
            # Check file structure
            if 'data' not in f_in.keys():
                print(f"Error: 'data' group not found in file {input_file}")
                return False

            data_group = f_in['data']

            # Get all demo names
            demo_names = [
                key for key in data_group.keys() if key.startswith('demo_')
            ]
            demo_names.sort()  # Ensure consistent order

            if not demo_names:
                print(f'Error: No demo data found in file {input_file}')
                return False

            total_demos = len(demo_names)
            num_samples = max(1, int(total_demos * sample_ratio))

            print(f'  Total demos: {total_demos}')
            print(f'  Sampling ratio: {sample_ratio:.1%}')
            print(f'  Sample count: {num_samples}')

            # Randomly select demos
            selected_demos = random.sample(demo_names, num_samples)
            selected_demos.sort()  # Keep sorted for readability

            print(
                f"  Selected demos: {selected_demos[:5]}{'...' if len(selected_demos) > 5 else ''}",
            )

            # Create output directory
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create output file and copy data
            with h5py.File(output_file, 'w') as f_out:
                # Create data group
                data_group_out = f_out.create_group('data')

                # Copy all attributes from data group
                for key, value in data_group.attrs.items():
                    data_group_out.attrs[key] = value

                # Copy selected demos
                total_samples = 0
                for i, demo_name in enumerate(selected_demos):
                    # Create new demo group (renumbered)
                    new_demo_name = f'demo_{i}'
                    demo_group_out = data_group_out.create_group(new_demo_name)

                    # Copy all data from demo group
                    demo_group_in = data_group[demo_name]
                    copy_hdf5_group(demo_group_in, demo_group_out)

                    # Accumulate sample count
                    if 'num_samples' in demo_group_in.attrs:
                        total_samples += demo_group_in.attrs['num_samples']
                    elif 'obs' in demo_group_in:
                        # If no num_samples attribute, try to infer from obs
                        obs_group = demo_group_in['obs']
                        # Find any dataset to infer length
                        for key in obs_group.keys():
                            if isinstance(obs_group[key], h5py.Dataset):
                                total_samples += len(obs_group[key])
                                break

                # Update statistics
                if 'num_demos' in data_group_out.attrs:
                    data_group_out.attrs['num_demos'] = num_samples
                if 'total' in data_group_out.attrs:
                    data_group_out.attrs['total'] = total_samples

                print(f'  Output file: {output_file}')
                print(f'  Retained demos: {num_samples}')
                print(f'  Total samples: {total_samples}')

            return True

    except Exception as e:
        print(f'Error processing file {input_file}: {e}')
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Randomly sample a certain proportion of data from HDF5 files and create new HDF5 files',
    )
    parser.add_argument('--input-file', type=str, help='Input HDF5 file path')
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output HDF5 file path (default: add _sampled suffix to input filename)',
    )
    parser.add_argument(
        '--ratio',
        type=float,
        required=True,
        help='Sampling ratio (0.0 - 1.0), e.g., 0.5 means sample 50%%',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory, batch process all HDF5 files in the directory',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory, used together with --input-dir',
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.hdf5',
        help='Filename pattern (default: *.hdf5)',
    )
    parser.add_argument(
        '--not-recursive',
        action='store_true',
        help='Do not recursively search subdirectories',
    )

    args = parser.parse_args()

    # Validate sampling ratio
    if args.ratio < 0.0 or args.ratio > 1.0:
        print('Error: Sampling ratio must be between 0.0 and 1.0')
        return

    # Batch processing mode
    if args.input_dir:
        if not args.output_dir:
            print(
                'Error: --output-dir must be specified when using --input-dir'
            )
            return

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        # Find all HDF5 files
        if args.not_recursive:
            demo_files = list(input_dir.glob(args.pattern))
        else:
            demo_files = list(input_dir.rglob(args.pattern))

        if not demo_files:
            print(
                f'No files matching {args.pattern} found in {args.input_dir}'
            )
            return

        print(f'Found {len(demo_files)} files to process\n')

        success_count = 0
        for demo_file in demo_files:
            # Generate output file path
            relative_path = demo_file.relative_to(input_dir)
            output_file = output_dir / relative_path

            # If output filename is same as input, add suffix
            if output_file == demo_file:
                output_file = (
                    output_file.parent
                    / f'{output_file.stem}_sampled{output_file.suffix}'
                )

            output_file.parent.mkdir(parents=True, exist_ok=True)

            if sample_hdf5_file(
                str(demo_file), str(output_file), args.ratio, args.seed
            ):
                success_count += 1
            print()

        print(
            f'Processing complete: {success_count}/{len(demo_files)} files succeeded'
        )

    # Single file processing mode
    else:
        if not args.input_file:
            print('Error: Must specify --input-file or --input-dir')
            return

        # Determine output file path
        if args.output_file:
            output_file = args.output_file
        else:
            input_path = Path(args.input_file)
            output_file = str(
                input_path.parent
                / f'{input_path.stem}_sampled{input_path.suffix}'
            )

        success = sample_hdf5_file(
            args.input_file, output_file, args.ratio, args.seed
        )
        if success:
            print('\nProcessing complete!')
        else:
            print('\nProcessing failed!')


if __name__ == '__main__':
    main()
