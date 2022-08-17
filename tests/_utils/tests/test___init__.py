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
"""Consolidate test configuration from command-line and environment.

"""
from __future__ import annotations

from .. import (
    DEFAULT_CPUS_PER_NODE,
    DEFAULT_GPU_DELAY,
    DEFAULT_GPU_MEMORY_BUDGET,
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_OMPS_PER_NODE,
    DEFAULT_OMPTHREADS,
    DEFAULT_PROCESS_ENV,
    FEATURES,
    PER_FILE_ARGS,
    SKIPPED_EXAMPLES,
    UI_WIDTH,
)


class TestConsts:
    def test_DEFAULT_CPUS_PER_NODE(self) -> None:
        assert DEFAULT_CPUS_PER_NODE == 4

    def test_DEFAULT_GPUS_PER_NODE(self) -> None:
        assert DEFAULT_GPUS_PER_NODE == 1

    def test_DEFAULT_GPU_DELAY(self) -> None:
        assert DEFAULT_GPU_DELAY == 2000

    def test_DEFAULT_GPU_MEMORY_BUDGET(self) -> None:
        assert DEFAULT_GPU_MEMORY_BUDGET == 4096

    def test_DEFAULT_OMPS_PER_NODE(self) -> None:
        assert DEFAULT_OMPS_PER_NODE == 1

    def test_DEFAULT_OMPTHREADS(self) -> None:
        assert DEFAULT_OMPTHREADS == 4

    def test_DEFAULT_PROCESS_ENV(self) -> None:
        assert DEFAULT_PROCESS_ENV == {
            "LEGATE_TEST": "1",
        }

    def test_UI_WIDTH(self) -> None:
        assert UI_WIDTH == 65

    def test_FEATURES(self) -> None:
        assert FEATURES == ("cpus", "cuda", "eager", "openmp")

    def test_SKIPPED_EXAMPLES(self) -> None:
        assert isinstance(SKIPPED_EXAMPLES, set)
        assert all(isinstance(x, str) for x in SKIPPED_EXAMPLES)
        assert all(x.startswith("examples") for x in SKIPPED_EXAMPLES)

    def test_PER_FILE_ARGS(self) -> None:
        assert isinstance(PER_FILE_ARGS, dict)
        assert all(isinstance(x, str) for x in PER_FILE_ARGS.keys())
        assert all(isinstance(x, list) for x in PER_FILE_ARGS.values())
