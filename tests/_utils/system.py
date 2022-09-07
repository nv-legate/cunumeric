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
"""Provide a System class to encapsulate process execution and reporting
system information (number of CPUs present, etc).

"""
from __future__ import annotations

import multiprocessing
import os
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from subprocess import PIPE, STDOUT, run as stdlib_run
from typing import Sequence

from .types import CPUInfo, EnvDict, GPUInfo


@dataclass
class ProcessResult:

    #: The command invovation, including relevant environment vars
    invocation: str

    #  User-friendly test file path to use in reported output
    test_file: Path

    #: Whether this process was actually invoked
    skipped: bool = False

    #: The returncode from the process
    returncode: int = 0

    #: The collected stdout and stderr output from the process
    output: str = ""


class System:
    """A facade class for system-related functions.

    Parameters
    ----------
    dry_run : bool, optional
        If True, no commands will be executed, but a log of any commands
        submitted to ``run`` will be made. (default: False)

    """

    def __init__(
        self,
        *,
        dry_run: bool = False,
    ) -> None:
        self.manager = multiprocessing.Manager()
        self.dry_run: bool = dry_run

    def run(
        self,
        cmd: Sequence[str],
        test_file: Path,
        *,
        env: EnvDict | None = None,
        cwd: str | None = None,
    ) -> ProcessResult:
        """Wrapper for subprocess.run that encapsulates logging.

        Parameters
        ----------
        cmd : sequence of str
            The command to run, split on whitespace into a sequence
            of strings

        test_file : Path
            User-friendly test file path to use in reported output

        env : dict[str, str] or None, optional, default: None
            Environment variables to apply when running the command

        cwd: str or None, optional, default: None
            A current working directory to pass to stdlib ``run``.

        """

        env = env or {}

        envstr = (
            " ".join(f"{k}={v}" for k, v in env.items())
            + min(len(env), 1) * " "
        )

        invocation = envstr + " ".join(cmd)

        if self.dry_run:
            return ProcessResult(invocation, test_file, skipped=True)

        full_env = dict(os.environ)
        full_env.update(env)

        proc = stdlib_run(
            cmd, cwd=cwd, env=full_env, stdout=PIPE, stderr=STDOUT, text=True
        )

        return ProcessResult(
            invocation,
            test_file,
            returncode=proc.returncode,
            output=proc.stdout,
        )

    @cached_property
    def cpus(self) -> tuple[CPUInfo, ...]:
        """A list of CPUs on the system."""

        N = multiprocessing.cpu_count()

        if sys.platform == "darwin":
            return tuple(CPUInfo((i,)) for i in range(N))

        sibling_sets: set[tuple[int, ...]] = set()
        for i in range(N):
            line = open(
                f"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list"
            ).read()
            sibling_sets.add(
                tuple(sorted(int(x) for x in line.strip().split(",")))
            )
        return tuple(CPUInfo(siblings) for siblings in sorted(sibling_sets))

    @cached_property
    def gpus(self) -> tuple[GPUInfo, ...]:
        """A list of GPUs on the system, including total memory information."""

        try:
            # This pynvml import is protected inside this method so that in
            # case pynvml is not installed, tests stages that don't need gpu
            # info (e.g. cpus, eager) will proceed unaffected. Test stages
            # that do require gpu info will fail here with an ImportError.
            import pynvml  # type: ignore[import]

            # Also a pynvml package is available on some platforms that won't
            # have GPUs for some reason. In which case this init call will
            # fail.
            pynvml.nvmlInit()
        except Exception:
            return ()

        num_gpus = pynvml.nvmlDeviceGetCount()

        results = []
        for i in range(num_gpus):
            info = pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByIndex(i)
            )
            results.append(GPUInfo(i, info.total))

        return tuple(results)
