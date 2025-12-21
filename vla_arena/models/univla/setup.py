#!/usr/bin/env python
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

import re
from os import path as op

from setuptools import find_packages, setup


def _read(f):
    return (
        open(op.join(op.dirname(__file__), f)).read() if op.exists(f) else ''
    )


_meta = _read('prismatic/__init__.py')


def find_meta(_meta, string):
    l_match = re.search(r'^' + string + r'\s*=\s*"(.*)"', _meta, re.M)
    if l_match:
        return l_match.group(1)
    raise RuntimeError(f'Unable to find {string} string.')


# install_requires = [
#     l for l in _read("requirements.txt").split("\n") if l and not l.startswith("#") and not l.startswith("-")
# ]

meta = dict(
    name=find_meta(_meta, '__project__'),
    version=find_meta(_meta, '__version__'),
    license=find_meta(_meta, '__license__'),
    description='UniVLA',
    platforms=('Any'),
    zip_safe=False,
    author=find_meta(_meta, '__author__'),
    author_email=find_meta(_meta, '__email__'),
    url='https://github.com/OpenDriveLab/UniVLA',
    packages=find_packages(exclude=['tests']),
    # install_requires=install_requires,
)

if __name__ == '__main__':
    print('find_package', find_packages(exclude=['tests']))
    setup(**meta)
