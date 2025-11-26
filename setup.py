# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "./README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="vla-arena",
    packages=[package for package in find_packages(where="./") if package.startswith("vla_arena")],
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="VLA-Arena: A Comprehensive Benchmark for Vision-Language-Action Models in Robotic Manipulation",
    author="Borong Zhang, Jiahao Li, Jiachen Shen",
    author_email="jiahaoli2077@gmail.com",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "vla-arena.main=vla_arena.main:main",
            "vla-arena.eval=vla_arena.evaluate:main",
            "vla-arena.config_copy=scripts.config_copy:main",
            "vla-arena.create_template=scripts.create_template:main",
        ]
    },
)
