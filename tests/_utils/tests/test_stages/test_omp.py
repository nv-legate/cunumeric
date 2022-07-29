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

from ...config import Config
from ...system import System
from ...test_stages import omp as m
from ...types import CPUInfo


class FakeSystem(System):
    @property
    def cpus(self) -> tuple[CPUInfo, ...]:
        return tuple(CPUInfo(i) for i in range(12))


def test_default() -> None:
    c = Config([])
    s = FakeSystem()
    stage = m.OMP(c, s)
    assert stage.kind == "openmp"
    assert stage.args == ["-cunumeric:test"]
    assert stage.env == {}
    assert stage.spec.workers > 0


@pytest.mark.parametrize("shard,expected", [[(2,), "2"], [(1, 2, 3), "1,2,3"]])
def test_shard_args(shard: tuple[int, ...], expected: str) -> None:
    c = Config([])
    s = FakeSystem()
    stage = m.OMP(c, s)
    result = stage.shard_args(shard, c)
    assert result == [
        "--omps",
        f"{c.omps}",
        "--ompthreads",
        f"{c.ompthreads}",
        "--cpu-bind",
        expected,
    ]


def test_spec_with_omps_1_threads_1() -> None:
    c = Config(["test.py", "--omps", "1", "--ompthreads", "1"])
    s = FakeSystem()
    stage = m.OMP(c, s)
    assert stage.spec.workers == 6
    assert stage.spec.shards == [(0,), (1,), (2,), (3,), (4,), (5,)]


def test_spec_with_omps_1_threads_2() -> None:
    c = Config(["test.py", "--omps", "1", "--ompthreads", "2"])
    s = FakeSystem()
    stage = m.OMP(c, s)
    assert stage.spec.workers == 4
    assert stage.spec.shards == [(0, 1), (2, 3), (4, 5), (6, 7)]


def test_spec_with_omps_2_threads_1() -> None:
    c = Config(["test.py", "--omps", "2", "--ompthreads", "1"])
    s = FakeSystem()
    stage = m.OMP(c, s)
    assert stage.spec.workers == 4
    assert stage.spec.shards == [(0, 1), (2, 3), (4, 5), (6, 7)]


def test_spec_with_omps_2_threads_2() -> None:
    c = Config(["test.py", "--omps", "2", "--ompthreads", "2"])
    s = FakeSystem()
    stage = m.OMP(c, s)
    assert stage.spec.workers == 2
    assert stage.spec.shards == [(0, 1, 2, 3), (4, 5, 6, 7)]


def test_spec_with_utility() -> None:
    c = Config(
        ["test.py", "--omps", "2", "--ompthreads", "2", "--utility", "3"]
    )
    s = FakeSystem()
    stage = m.OMP(c, s)
    assert stage.spec.workers == 1
    assert stage.spec.shards == [(0, 1, 2, 3)]


def test_spec_with_requested_workers() -> None:
    c = Config(["test.py", "--omps", "1", "--ompthreads", "1", "-j", "2"])
    s = FakeSystem()
    stage = m.OMP(c, s)
    assert stage.spec.workers == 2
    assert stage.spec.shards == [(0,), (1,)]


def test_spec_with_requested_workers_zero() -> None:
    s = FakeSystem()
    c = Config(["test.py", "-j", "0"])
    assert c.requested_workers == 0
    with pytest.raises(RuntimeError):
        m.OMP(c, s)


def test_spec_with_requested_workers_bad() -> None:
    s = FakeSystem()
    c = Config(["test.py", "-j", f"{len(s.cpus)+1}"])
    assert c.requested_workers > len(s.cpus)
    with pytest.raises(RuntimeError):
        m.OMP(c, s)


def test_spec_with_verbose() -> None:
    c = Config(["test.py", "--verbose", "--cpus", "2"])
    s = FakeSystem()
    stage = m.OMP(c, s)
    assert stage.spec.workers == 1
    assert stage.spec.shards == [(0, 1, 2, 3)]
