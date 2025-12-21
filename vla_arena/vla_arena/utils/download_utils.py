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
Download functionalities adapted from Mandlekar et. al.: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/file_utils.py
"""
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from termcolor import colored
from tqdm import tqdm

from vla_arena.vla_arena import get_vla_arena_path


try:
    import shutil

    from huggingface_hub import snapshot_download

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


class DownloadProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    From https://gist.github.com/dehowell/884204.
    Args:
        url (str): url string
    Returns:
        is_alive (bool): True if url is reachable, False otherwise
    """
    request = urllib.request.Request(url)
    # request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def download_url(url, download_dir, check_overwrite=True, is_zipfile=True):
    """
    First checks that @url is reachable, then downloads the file
    at that url into the directory specified by @download_dir.
    Prints a progress bar during the download using tqdm.
    Modified from https://github.com/tqdm/tqdm#hooks-and-callbacks, and
    https://stackoverflow.com/a/53877507.
    Args:
        url (str): url string
        download_dir (str): path to directory where file should be downloaded
        check_overwrite (bool): if True, will sanity check the download fpath to make sure a file of that name
            doesn't already exist there
    """

    # check if url is reachable. We need the sleep to make sure server doesn't reject subsequent requests
    assert url_is_alive(url), f'@download_url got unreachable url: {url}'
    time.sleep(0.5)

    # infer filename from url link
    fname = url.split('/')[-1]
    file_to_write = os.path.join(download_dir, fname)

    # If we're checking overwrite and the path already exists,
    # we ask the user to verify that they want to overwrite the file
    user_response = None
    if check_overwrite and os.path.exists(file_to_write):
        user_response = input(
            f'Warning: file {file_to_write} already exists. Overwrite? y/n\n'
        )
        # assert user_response.lower() in {"yes", "y"}, f"Did not receive confirmation. Aborting download."

    if user_response is None or user_response.lower() in {'yes', 'y'}:
        with DownloadProgressBar(
            unit='B', unit_scale=True, miniters=1, desc=fname
        ) as t:
            urllib.request.urlretrieve(
                url, filename=file_to_write, reporthook=t.update_to
            )
    if is_zipfile:
        with zipfile.ZipFile(file_to_write, 'r') as archive:
            archive.extractall(path=download_dir)
        if os.path.isfile(file_to_write):
            os.remove(file_to_write)


DATASET_LINKS = {
    'vla_arena_object': 'https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip',
    'vla_arena_goal': 'https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip',
    'vla_arena_spatial': 'https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip',
    'vla_arena_100': 'https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip',
}

HF_REPO_ID = 'yifengzhu-hf/LIBERO-datasets'


def download_from_huggingface(
    dataset_name, download_dir, check_overwrite=True
):
    """
    Download a specific LIBERO dataset from Hugging Face.

    Args:
        dataset_name (str): Name of the dataset to download (e.g., 'vla_arena_spatial')
        download_dir (str): Directory where the dataset should be downloaded
        check_overwrite (bool): If True, will check if dataset already exists
    """
    if not HUGGINGFACE_AVAILABLE:
        raise ImportError(
            "Hugging Face Hub is not available. Install it with 'pip install huggingface_hub'",
        )

    # Create the destination folder
    os.makedirs(download_dir, exist_ok=True)

    # Check if dataset already exists
    dataset_dir = os.path.join(download_dir, dataset_name)
    if check_overwrite and os.path.exists(dataset_dir):
        user_response = input(
            f'Warning: dataset {dataset_name} already exists at {dataset_dir}. Overwrite? y/n\n',
        )
        if user_response.lower() not in {'yes', 'y'}:
            print(f'Skipping download of {dataset_name}')
            return

        # Remove existing directory
        print(f'Removing existing folder: {dataset_dir}')
        shutil.rmtree(dataset_dir)

    # Download the dataset
    print(f'Downloading {dataset_name} from Hugging Face...')
    folder_path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type='dataset',
        local_dir=download_dir,
        allow_patterns=f'{dataset_name}/*',
        local_dir_use_symlinks=False,  # Prevents using symlinks to cached files
        force_download=True,  # Forces re-downloading files
    )

    # Verify downloaded files
    file_count = sum(
        [
            len(files)
            for _, _, files in os.walk(
                os.path.join(download_dir, dataset_name)
            )
        ],
    )
    print(f'Downloaded {file_count} files for {dataset_name}')


def vla_arena_dataset_download(
    datasets='all',
    download_dir=None,
    check_overwrite=True,
    use_huggingface=False,
):
    """Download vla_arena datasets

    Args:
        datasets (str, optional): Specify which datasets to save. Defaults to "all", downloading all the datasets.
        download_dir (str, optional): Target location for storing datasets. Defaults to None, using the default path.
        check_overwrite (bool, optional): Check if overwriting datasets. Defaults to True.
        use_huggingface (bool, optional): Use Hugging Face instead of the original download links. Defaults to False.
    """
    if download_dir is None:
        download_dir = get_vla_arena_path('datasets')
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    assert datasets in [
        'all',
        'vla_arena_object',
        'vla_arena_goal',
        'vla_arena_spatial',
        'vla_arena_100',
    ]

    datasets_to_download = (
        [
            'vla_arena_object',
            'vla_arena_goal',
            'vla_arena_spatial',
            'vla_arena_100',
        ]
        if datasets == 'all'
        else [datasets]
    )

    for dataset_name in datasets_to_download:
        print(f'Downloading {dataset_name}')

        if use_huggingface:
            download_from_huggingface(
                dataset_name=dataset_name,
                download_dir=download_dir,
                check_overwrite=check_overwrite,
            )
        else:
            print('Using original download links (these may expire soon)')
            download_url(
                DATASET_LINKS[dataset_name],
                download_dir=download_dir,
                check_overwrite=check_overwrite,
            )


def check_vla_arena_dataset(download_dir=None):
    """Check the integrity of the downloaded datasets.

    Args:
        download_dir (str, optional): The path where datasets are stored. Defaults to None, using the default path.

    Returns:
        bool: True if the datasets are successfully downloaded, False otherwise.
    """
    if download_dir is None:
        download_dir = get_vla_arena_path('datasets')
    check_result = True
    for dataset_name in [
        'vla_arena_object',
        'vla_arena_goal',
        'vla_arena_spatial',
        'vla_arena_10',
        'vla_arena_90',
    ]:
        info_str = ''
        dataset_status = False
        dataset_dir = os.path.join(download_dir, dataset_name)
        if os.path.exists(dataset_dir):
            count = 0
            for path in Path(dataset_dir).glob('*.hdf5'):
                count += 1
            if (count == 10 and dataset_name != 'vla_arena_90') or (
                count == 90 and dataset_name == 'vla_arena_90'
            ):
                dataset_status = True
                info_str = colored(
                    f'[X] Dataset {dataset_name} is complete',
                    'green',
                    attrs=['bold'],
                )
            else:
                colored(
                    f'[?] Dataset {dataset_name} is not downloaded completely',
                    'yellow',
                    attrs=['bold'],
                )
        else:
            info_str = colored(
                f'[ ] Dataset {dataset_name} not found!!!',
                'red',
                attrs=['bold'],
            )

        print(info_str)
        check_result = check_result and dataset_status
    return check_result
