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

import h5py


def print_dataset_info(name, obj):
    """Callback function to print information about HDF5 objects."""
    indent_level = name.count('/')
    indent = '  ' * indent_level

    if isinstance(obj, h5py.Dataset):
        # Print dataset information
        shape = obj.shape
        dtype = obj.dtype
        print(f'{indent}- Dataset: {name} | Shape: {shape} | Type: {dtype}')

        # Try to show first few data points
        try:
            data_preview = obj[...]
            if data_preview.size > 0:
                # Limit display count to avoid excessive output
                preview_flat = data_preview.flatten()
                preview_size = min(5, preview_flat.size)
                preview_str = ', '.join(
                    str(x) for x in preview_flat[:preview_size]
                )
                print(
                    f"{indent}    Sample data: {preview_str}{' ...' if preview_flat.size > preview_size else ''}",
                )
        except Exception:
            print(f'{indent}    (Unable to read data sample)')

        # Print attributes
        if obj.attrs:
            print(f'{indent}    Attributes:')
            for key, value in obj.attrs.items():
                print(f'{indent}      - {key}: {value}')

    elif isinstance(obj, h5py.Group):
        # Print group information
        print(f"{indent}+ Group: {name if name else '/'}")
        if obj.attrs:
            print(f'{indent}    Attributes:')
            for key, value in obj.attrs.items():
                print(f'{indent}      - {key}: {value}')


def inspect_hdf5(file_path, dataset_path=None):
    """Inspect HDF5 file structure and content samples."""
    print(f'Checking file: {file_path}')

    with h5py.File(file_path, 'r') as h5_file:
        if dataset_path:
            if dataset_path in h5_file:
                obj = h5_file[dataset_path]
                print_dataset_info(dataset_path, obj)
            else:
                print(
                    f'Path {dataset_path} does not exist. Available keys include:'
                )
                for key in h5_file.keys():
                    print(f'- {key}')
        else:
            h5_file.visititems(print_dataset_info)


def main():
    parser = argparse.ArgumentParser(
        description='Print keys and value samples from HDF5 file'
    )
    parser.add_argument('file', type=str, help='HDF5 file path')
    parser.add_argument(
        '--path',
        type=str,
        default=None,
        help='Specify dataset path to view, default prints entire file structure',
    )

    args = parser.parse_args()
    inspect_hdf5(args.file, args.path)


if __name__ == '__main__':
    main()
