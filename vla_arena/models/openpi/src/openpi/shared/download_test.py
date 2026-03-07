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

import pathlib

import openpi.shared.download as download
import pytest


@pytest.fixture(scope='session', autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp('openpi_data')
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv('OPENPI_DATA_HOME', str(temp_dir))
        yield


def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / 'local'
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download('bogus')


def test_download_gs_dir():
    remote_path = 'gs://openpi-assets/testdata/random'

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_gs():
    remote_path = 'gs://openpi-assets/testdata/random/random_512kb.bin'

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_fsspec():
    remote_path = 'gs://big_vision/paligemma_tokenizer.model'

    local_path = download.maybe_download(remote_path, gs={'token': 'anon'})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={'token': 'anon'})
    assert new_local_path == local_path
