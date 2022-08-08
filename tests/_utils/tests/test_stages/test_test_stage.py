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

import pytest

from ...test_stages import eager as m


class Test_adjust_workers:
    @pytest.mark.parametrize("n", (1, 5, 100))
    def test_None_requested(self, n: int) -> None:
        assert m.adjust_workers(n, None) == n

    @pytest.mark.parametrize("n", (1, 2, 9))
    def test_requested(self, n: int) -> None:
        assert m.adjust_workers(10, n) == n

    def test_negative_requested(self) -> None:
        with pytest.raises(ValueError):
            assert m.adjust_workers(10, -1)

    def test_zero_requested(self) -> None:
        with pytest.raises(RuntimeError):
            assert m.adjust_workers(10, 0)

    def test_zero_computed(self) -> None:
        with pytest.raises(RuntimeError):
            assert m.adjust_workers(0, None)

    def test_requested_too_large(self) -> None:
        with pytest.raises(RuntimeError):
            assert m.adjust_workers(10, 11)
