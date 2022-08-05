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
from ...test_stages import gpu as m
from ...types import GPUInfo


class FakeSystem(System):
    @property
    def gpus(self) -> tuple[GPUInfo, ...]:
        return tuple(GPUInfo(i, 6 << 32) for i in range(6))


def test_default() -> None:
    c = Config([])
    s = FakeSystem()
    stage = m.GPU(c, s)
    assert stage.kind == "cuda"
    assert stage.args == ["-cunumeric:test"]
    assert stage.env == {}
    assert stage.spec.workers > 0


@pytest.mark.parametrize("shard,expected", [[(2,), "2"], [(1, 2, 3), "1,2,3"]])
def test_shard_args(shard: tuple[int, ...], expected: str) -> None:
    c = Config([])
    s = FakeSystem()
    stage = m.GPU(c, s)
    result = stage.shard_args(shard, c)
    assert result == ["--gpus", f"{len(shard)}", "--gpu-bind", expected]


def test_spec_with_gpus_1() -> None:
    c = Config(["test.py", "--gpus", "1"])
    s = FakeSystem()
    stage = m.GPU(c, s)
    assert stage.spec.workers == 24
    assert stage.spec.shards == [(0,), (1,), (2,), (3,), (4,), (5,)] * 24


def test_spec_with_gpus_2() -> None:
    c = Config(["test.py", "--gpus", "2"])
    s = FakeSystem()
    stage = m.GPU(c, s)
    assert stage.spec.workers == 12
    assert stage.spec.shards == [(0, 1), (2, 3), (4, 5)] * 12


def test_spec_with_requested_workers() -> None:
    c = Config(["test.py", "--gpus", "1", "-j", "2"])
    s = FakeSystem()
    stage = m.GPU(c, s)
    assert stage.spec.workers == 2
    assert stage.spec.shards == [(0,), (1,), (2,), (3,), (4,), (5,)] * 2


def test_spec_with_requested_workers_zero() -> None:
    s = FakeSystem()
    c = Config(["test.py", "-j", "0"])
    assert c.requested_workers == 0
    with pytest.raises(RuntimeError):
        m.GPU(c, s)


def test_spec_with_requested_workers_bad() -> None:
    s = FakeSystem()
    c = Config(["test.py", "-j", f"{len(s.gpus)+100}"])
    assert c.requested_workers > len(s.gpus)
    with pytest.raises(RuntimeError):
        m.GPU(c, s)


def test_spec_with_verbose() -> None:
    args = ["test.py", "--gpus", "2"]
    c = Config(args)
    cv = Config(args + ["--verbose"])
    s = FakeSystem()

    spec, vspec = m.GPU(c, s).spec, m.GPU(cv, s).spec
    assert vspec == spec
