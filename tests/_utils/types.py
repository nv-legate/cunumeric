# Copyright 2022 NVIDIA Corporation
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
"""Provide types that are useful throughout the test driver code.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from typing_extensions import TypeAlias


@dataclass(frozen=True)
class CPUInfo:
    """Encapsulate information about a single CPU"""

    #: ID of the CPU to specify in test shards
    id: int


@dataclass(frozen=True)
class GPUInfo:
    """Encapsulate information about a single CPU"""

    #: ID of the GPU to specify in test shards
    id: int

    #: The totl framebuffer memory of this GPU
    total: int


#: Represent command line arguments
ArgList = List[str]


#: Represent str->str environment variable mappings
EnvDict: TypeAlias = Dict[str, str]
