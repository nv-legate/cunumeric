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

from pathlib import Path
from subprocess import CompletedProcess
from typing import Iterator

from ..config import Config
from ..system import ArgList, System
from .test_stage import TestStage


class CPU(TestStage):
    """A test stage for exercising CPU features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: System
        Process execution wrapper

    """

    kind = "cpus"

    def __init__(self, config: Config, system: System) -> None:
        self.workers = self.compute_workers(config, system)

    def run(
        self, test_file: Path, config: Config, system: System
    ) -> CompletedProcess[str]:
        test_path = config.root_dir / test_file
        stage_args = ["-cunumeric:test"] + next(self.cpu_args(config))
        file_args = self.file_args(test_file, config)

        cmd = ["legate", str(test_path)]
        cmd += stage_args + file_args + config.extra_args

        result = system.run(cmd, env=system.env)
        self._log_proc(result, test_file, config.verbose)
        return result

    def cpu_args(self, config: Config) -> Iterator[ArgList]:
        yield ["--cpus", str(config.cpus)]

    def compute_workers(self, config: Config, system: System) -> int:
        if config.requested_workers is not None:
            return config.requested_workers

        return 1 if config.verbose else len(system.cpus) // config.cpus
