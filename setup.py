#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
__author__ = "Randall Balestriero"

import versioneer

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="symjax",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Randall Balestriero",
    author_email="randallbalestriero@gmail.com",
    description="A Symbolic JAX software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SymJAX/SymJAX.git",
    packages=find_packages(exclude=["examples"]),
    classifiers=[
        "Natural Language :: English",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "scipy", "jax", "jaxlib", "networkx",],
    license="Apache-2.0",
)
