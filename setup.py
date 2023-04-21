#!/usr/bin/env python3

# Copyright 2021-2022 NVIDIA Corporation
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
#

from setuptools import find_packages
from skbuild import setup

import versioneer

setup(
    name="cunumeric",
    version=versioneer.get_version(),
    description="An Aspiring Drop-In Replacement for NumPy at Scale",
    url="https://github.com/nv-legate/cunumeric",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(
        where=".",
        include=["cunumeric*"],
    ),
    package_data={"cunumeric": ["_sphinxext/_templates/*.rst"]},
    include_package_data=True,
    cmdclass=versioneer.get_cmdclass(),
    install_requires=["numpy>=1.22"],
    zip_safe=False,
)
