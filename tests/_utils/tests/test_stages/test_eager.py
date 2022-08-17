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
from ...test_stages._linux import eager as m
from ...types import CPUInfo


class FakeSystem(System):
    @property
    def cpus(self) -> tuple[CPUInfo, ...]:
        return tuple(CPUInfo(i) for i in range(6))


def test_default() -> None:
    c = Config([])
    s = FakeSystem()
    stage = m.Eager(c, s)
    assert stage.kind == "eager"
    assert stage.args == []
    assert stage.env == {
        "CUNUMERIC_MIN_CPU_CHUNK": "2000000000",
        "CUNUMERIC_MIN_OMP_CHUNK": "2000000000",
        "CUNUMERIC_MIN_GPU_CHUNK": "2000000000",
    }
    assert stage.spec.workers > 0


@pytest.mark.parametrize("shard,expected", [[(2,), "2"], [(1, 2, 3), "1,2,3"]])
def test_shard_args(shard: tuple[int, ...], expected: str) -> None:
    c = Config([])
    s = FakeSystem()
    stage = m.Eager(c, s)
    result = stage.shard_args(shard, c)
    assert result == ["--cpus", "1", "--cpu-bind", expected]


def test_spec() -> None:
    c = Config([])
    s = FakeSystem()
    stage = m.Eager(c, s)
    assert stage.spec.workers == len(s.cpus)
    assert stage.spec.shards == [tuple([i]) for i in range(stage.spec.workers)]


def test_spec_with_requested_workers_zero() -> None:
    s = FakeSystem()
    c = Config(["test.py", "-j", "0"])
    assert c.requested_workers == 0
    with pytest.raises(RuntimeError):
        m.Eager(c, s)


def test_spec_with_requested_workers_bad() -> None:
    s = FakeSystem()
    c = Config(["test.py", "-j", f"{len(s.cpus)+1}"])
    assert c.requested_workers > len(s.cpus)
    with pytest.raises(RuntimeError):
        m.Eager(c, s)


def test_spec_with_verbose() -> None:
    c = Config(["test.py"])
    cv = Config(["test.py", "--verbose"])
    s = FakeSystem()

    spec, vspec = m.Eager(c, s).spec, m.Eager(cv, s).spec
    assert vspec == spec
