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

# read the contents of your README file
from os import path

from setuptools import find_packages, setup


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, './README.md'), encoding='utf-8') as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if '.png' not in x]
long_description = ''.join(lines)

setup(
    name='vla-arena',
    packages=[
        package
        for package in find_packages()
        if package.startswith('vla_arena')
    ],
    install_requires=[],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    description='VLA-Arena: Benchmarking Vision-Language-Action Models by Structured Task Design',
    author='Borong Zhang, Jiahao Li, Jiachen Shen',
    author_email='jiahaoli2077@gmail.com',
    version='0.1.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'vla_arena.main=vla_arena.main:main',
            'vla_arena.eval=vla_arena.evaluate:main',
            'vla_arena.config_copy=scripts.config_copy:main',
            'vla_arena.create_template=scripts.create_template:main',
        ],
    },
)
