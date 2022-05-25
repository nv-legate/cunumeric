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

from itertools import chain

from .config import Config
from .logger import LOG
from .system import System
from .test_stages import STAGES
from .ui import DEFAULT_WIDTH, banner, bottom_line, yellow


class TestPlan:
    def __init__(self, config: Config, system: System) -> None:
        self._config = config
        self._system = system
        self._stages = [
            stage(config, system)
            for stage in STAGES
            if stage.kind in config.features
        ]

    def execute(self) -> int:
        LOG.clear()

        LOG(self.intro)

        for stage in self._stages:
            LOG(stage.intro)
            stage(self._config, self._system)
            LOG(stage.outro)

        all_procs = tuple(
            chain.from_iterable(s.result.procs for s in self._stages)
        )
        total = len(all_procs)
        passed = sum(proc.returncode == 0 for proc in all_procs)

        LOG(self.outro(total, passed))

        return int((total - passed) > 0)

    @property
    def intro(self) -> str:
        details = (
            f"* Feature stages       : {', '.join(yellow(x) for x in self._config.features)}",  # noqa E501
            f"* Test files per stage : {yellow(str(len(self._config.test_files)))}",  # noqa E501
        )
        return banner("Test Suite Configuration", details=details)

    def outro(self, total: int, passed: int) -> str:
        summary = bottom_line("All tests", total, passed)
        return f"\n{56*'~': >{DEFAULT_WIDTH}}\n{summary}\n"
