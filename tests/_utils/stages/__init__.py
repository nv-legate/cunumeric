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
"""Provide TestStage subclasses for running configured test files using
specific features.

"""
from __future__ import annotations

import sys
from typing import Dict, Type

from .. import FeatureType
from .test_stage import TestStage
from .util import log_proc

if sys.platform == "darwin":
    from ._osx import CPU, Eager, GPU, OMP
elif sys.platform.startswith("linux"):
    from ._linux import CPU, Eager, GPU, OMP
else:
    raise RuntimeError(f"unsupported platform: {sys.platform}")

#: All the available test stages that can be selected
STAGES: Dict[FeatureType, Type[TestStage]] = {
    "cpus": CPU,
    "cuda": GPU,
    "openmp": OMP,
    "eager": Eager,
}
