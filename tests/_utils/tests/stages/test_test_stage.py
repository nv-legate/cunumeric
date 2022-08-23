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

from datetime import timedelta
from pathlib import Path

from ... import FeatureType
from ...config import Config
from ...stages import test_stage as m
from ...stages.util import StageResult, StageSpec
from ...system import ProcessResult, System
from . import FakeSystem

s = FakeSystem()


class MockTestStage(m.TestStage):

    kind: FeatureType = "eager"

    name = "mock"

    args = ["-foo", "-bar"]

    def __init__(self, config: Config, system: System) -> None:
        self._init(config, system)

    def compute_spec(self, config: Config, system: System) -> StageSpec:
        return StageSpec(2, [(0,), (1,), (2,)])


class TestTestStage:
    def test_name(self) -> None:
        c = Config([])
        stage = MockTestStage(c, s)
        assert stage.name == "mock"

    def test_intro(self) -> None:
        c = Config([])
        stage = MockTestStage(c, s)
        assert "Entering stage: mock" in stage.intro

    def test_outro(self) -> None:
        c = Config([])
        stage = MockTestStage(c, s)
        stage.result = StageResult(
            [ProcessResult("invoke", Path("test/file"))],
            timedelta(seconds=2.123),
        )
        outro = stage.outro
        assert "Exiting stage: mock" in outro
        assert "Passed 1 of 1 tests (100.0%)" in outro
        assert "2.123" in outro

    def test_file_args_default(self) -> None:
        c = Config([])
        stage = MockTestStage(c, s)
        assert stage.file_args(Path("integration/foo"), c) == []
        assert stage.file_args(Path("unit/foo"), c) == []

    def test_file_args_v(self) -> None:
        c = Config(["test.py", "-v"])
        stage = MockTestStage(c, s)
        assert stage.file_args(Path("integration/foo"), c) == ["-v"]
        assert stage.file_args(Path("unit/foo"), c) == []

    def test_file_args_vv(self) -> None:
        c = Config(["test.py", "-vv"])
        stage = MockTestStage(c, s)
        assert stage.file_args(Path("integration/foo"), c) == ["-v", "-s"]
        assert stage.file_args(Path("unit/foo"), c) == []
