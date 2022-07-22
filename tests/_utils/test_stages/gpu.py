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

from .. import DEFAULT_GPU_MEMORY_BUDGET, DEFAULT_GPU_PARALLELISM
from ..config import Config
from ..system import ArgList, System
from .test_stage import TestStage


class GPU(TestStage):
    """A test stage for exercising GPU features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: System
        Process execution wrapper

    """

    kind = "cuda"

    def __init__(self, config: Config, system: System) -> None:
        self.workers = self.compute_workers(config, system)

    def run(
        self, test_file: Path, config: Config, system: System
    ) -> CompletedProcess[str]:
        test_path = config.root_dir / test_file
        stage_args = ["-cunumeric:test"] + next(self.gpu_args(config))
        file_args = self.file_args(test_file, config)

        cmd = [str(config.legate_path), str(test_path)]
        cmd += stage_args + file_args + config.extra_args

        result = system.run(cmd, env=system.env)
        self._log_proc(result, test_file, config.verbose)
        return result

    def gpu_args(self, config: Config) -> Iterator[ArgList]:
        yield ["--gpus", str(config.gpus)]

    def compute_workers(self, config: Config, system: System) -> int:

        gpus = system.gpus
        assert len(gpus)

        min_free = min(info.free for info in gpus)

        parallelism_per_gpu = min(
            DEFAULT_GPU_PARALLELISM, min_free // DEFAULT_GPU_MEMORY_BUDGET
        )

        return (
            parallelism_per_gpu
            * (len(gpus) // config.gpus)
            * config.gpus
            // len(gpus)
        )
