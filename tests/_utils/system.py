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

import multiprocessing
import os
from dataclasses import dataclass
from subprocess import PIPE, STDOUT, CompletedProcess, run as stdlib_run
from typing import Any, Dict, Sequence

from typing_extensions import TypeAlias

from .logger import LOG
from .ui import shell

SKIPPED_RETURNCODE = -99999


@dataclass(frozen=True)
class CPUInfo:
    id: int


@dataclass(frozen=True)
class GPUInfo:
    id: int
    free: int


EnvDict: TypeAlias = Dict[str, str]


class System:
    def __init__(self, *, dry_run: bool = False, debug: bool = False) -> None:
        self.dry_run: bool = dry_run
        self.debug = debug

    def run(
        self,
        cmd: Sequence[str],
        *,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
    ) -> CompletedProcess[str]:

        if self.dry_run or self.debug:
            LOG.record(shell(" ".join(cmd)))

        if self.dry_run:
            return CompletedProcess(cmd, SKIPPED_RETURNCODE, stdout="")

        return stdlib_run(
            cmd, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT, text=True
        )

    @property
    def cpus(self) -> tuple[CPUInfo, ...]:
        return tuple(CPUInfo(i) for i in range(multiprocessing.cpu_count()))

    @property
    def gpus(self) -> tuple[GPUInfo, ...]:
        try:
            import pynvml  # type: ignore[import]

            pynvml.nvmlInit()

            num_gpus = pynvml.nvmlDeviceGetCount()

            results = []
            for i in range(num_gpus):
                info = pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                )
                results.append(GPUInfo(i, info.free))
            return tuple(results)

        except ImportError:
            return ()

    @property
    def env(self) -> EnvDict:
        env = dict(os.environ)
        env["LEGATE_TEST"] = "1"
        env["REALM_SYNTHETIC_CORE_MAP"] = ""
        return env
