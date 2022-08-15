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
"""Utilities and helpers for implementing the Cunumeric custom test runner.

"""
from __future__ import annotations

from typing import Union
from typing_extensions import Literal, TypeAlias

#: Define the available feature types for tests
FeatureType: TypeAlias = Union[
    Literal["cpus"], Literal["cuda"], Literal["eager"], Literal["openmp"]
]

#: Value to use if --cpus is not specified.
DEFAULT_CPUS_PER_NODE = 4

#: Value to use if --gpus is not specified.
DEFAULT_GPUS_PER_NODE = 1

# Value to use if --fbmem is not specified (MB)
DEFAULT_GPU_MEMORY_BUDGET = 4096

#: Value to use if --omps is not specified.
DEFAULT_OMPS_PER_NODE = 1

#: Value to use if --ompthreads is not specified.
DEFAULT_OMPTHREADS = 4

#: Default values to apply to normalize the testing environment.
DEFAULT_PROCESS_ENV = {
    "LEGATE_TEST": "1",
}

#: Width for terminal ouput headers and footers.
UI_WIDTH = 65

#: Feature values that are accepted for --use, in the relative order
#: that the corresponding test stages should always execute in
FEATURES: tuple[FeatureType, ...] = (
    "cpus",
    "cuda",
    "eager",
    "openmp",
)

#: Paths to example files that should be skipped.
SKIPPED_EXAMPLES = {
    "examples/ingest.py",
    "examples/kmeans_sort.py",
    "examples/lstm_full.py",
    "examples/wgrad.py",
}

#: Extra arguments to supply when specific examples are executed.
PER_FILE_ARGS = {
    "examples/lstm_full.py": ["--file", "resources/lstm_input.txt"],
}
