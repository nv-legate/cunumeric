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
from __future__ import annotations

from typing import Union
from typing_extensions import Literal, TypeAlias

FeatureType: TypeAlias = Union[
    Literal["cpus"], Literal["cuda"], Literal["eager"], Literal["openmp"]
]

DEFAULT_CPUS_PER_NODE = 4

DEFAULT_GPUS_PER_NODE = 1

DEFAULT_GPU_MEMORY_BUDGET = 6 << 30

DEFAULT_GPU_PARALLELISM = 16

DEFAULT_OMPS_PER_NODE = 1

DEFAULT_OMPTHREADS = 1

UI_WIDTH = 60

FEATURES: set[FeatureType] = {
    "cpus",
    "cuda",
    "eager",
    "openmp",
}

SKIPPED_EXAMPLES = {
    "examples/ingest.py",
    "examples/kmeans_sort.py",
    "examples/lstm_full.py",
    "examples/wgrad.py",
}

PER_FILE_ARGS = {
    "examples/lstm_full.py": ["--file", "resources/lstm_input.txt"],
}
