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
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import CompletedProcess
from typing import Iterator, Type

from typing_extensions import Protocol

from . import PER_FILE_ARGS
from .config import Config
from .logger import LOG
from .system import SKIPPED_RETURNCODE, System
from .ui import banner, bottom_line, failed, passed, skipped, yellow


@dataclass(frozen=True)
class StageResult:
    procs: list[CompletedProcess[str]]
    time: timedelta


class TestStage(Protocol):

    kind: str

    result: StageResult

    def __call__(self, config: Config, system: System) -> None:

        t0 = datetime.now()
        procs = self._launch(config, system)
        t1 = datetime.now()

        self.result = StageResult(procs, t1 - t0)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def intro(self) -> str:
        return banner(f"Entering stage: {self.name} tests") + "\n"

    @property
    def outro(self) -> str:
        total = len(self.result.procs)
        passed = len([p for p in self.result.procs if p.returncode == 0])

        summary = bottom_line(self.name, total, passed)

        footer = banner(
            f"Exiting state: {self.name}",
            details=(
                "* Results      : "
                + yellow(
                    f"{passed} / {total} files passed ({passed/total*100:0.1f}%)"  # noqa E500
                ),
                "* Elapsed time : " + yellow(f"{self.result.time}"),
            ),
        )

        return f"{summary}\n{footer}"

    def run(
        self, test_file: Path, config: Config, system: System
    ) -> CompletedProcess[str]:
        ...

    def file_args(self, test_file: Path, config: Config) -> list[str]:
        test_file_string = str(test_file)
        args = PER_FILE_ARGS.get(test_file_string, [])

        # This is somewhat ugly but necessary in order to make pytest generate
        # more verbose output for integration tests when --verbose is specified
        if "integration" in test_file_string and config.verbose:
            args += ["-v"]

        return args

    def _launch(
        self, config: Config, system: System
    ) -> list[CompletedProcess[str]]:

        pool = multiprocessing.Pool(config.workers or 1)

        jobs = [
            pool.apply_async(self.run, (path, config, system))
            for path in config.test_files
        ]
        pool.close()

        return [job.get() for job in jobs]

    def _log_proc(
        self, proc: CompletedProcess[str], test_file: Path, verbose: bool
    ) -> None:
        msg = f"({self.name}) {test_file}"
        details = proc.stdout.split("\n") if verbose else None
        if proc.returncode == 0:
            LOG(passed(msg, details=details))
        elif proc.returncode == SKIPPED_RETURNCODE:
            LOG(skipped(msg))
        else:
            LOG(failed(msg, details=details))


class CPU(TestStage):

    kind = "cpus"

    def run(
        self, test_file: Path, config: Config, system: System
    ) -> CompletedProcess[str]:
        test_path = config.root_dir / test_file
        stage_args = ["-cunumeric:test"] + next(self.cpu_args(config))
        file_args = self.file_args(test_file, config)

        cmd = [str(config.legate_path), str(test_path)]
        cmd += stage_args + file_args + config.extra_args

        result = system.run(cmd, env=system.env)
        self._log_proc(result, test_file, config.verbose)
        return result

    def cpu_args(self, config: Config) -> Iterator[list[str]]:
        cpus = config.cpus or config.DEFAULT_CPUS
        yield ["--cpus", str(cpus)]


class GPU(TestStage):

    kind = "cuda"

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

    def gpu_args(self, config: Config) -> Iterator[list[str]]:
        gpus = config.gpus or config.DEFAULT_GPUS
        yield ["--gpus", str(gpus)]


class OMP(TestStage):

    kind = "openmp"

    def run(
        self, test_file: Path, config: Config, system: System
    ) -> CompletedProcess[str]:
        test_path = config.root_dir / test_file
        stage_args = ["-cunumeric:test"] + self.omp_args(config)
        file_args = self.file_args(test_file, config)

        cmd = [str(config.legate_path), str(test_path)]
        cmd += stage_args + file_args + config.extra_args

        result = system.run(cmd, env=system.env)
        self._log_proc(result, test_file, config.verbose)
        return result

    def omp_args(self, config: Config) -> list[str]:
        return [
            "--omps",
            str(config.omps),
            "--ompthreads",
            str(config.ompthreads),
        ]


class Eager(TestStage):

    kind = "eager"

    def run(
        self, test_file: Path, config: Config, system: System
    ) -> CompletedProcess[str]:
        test_path = config.root_dir / test_file
        stage_args = ["--cpus", "1"]
        file_args = self.file_args(test_file, config)

        cmd = [str(config.legate_path), str(test_path)]
        cmd += stage_args + file_args + config.extra_args

        env = system.env
        env.update(
            CUNUMERIC_MIN_CPU_CHUNK="2000000000",
            CUNUMERIC_MIN_OMP_CHUNK="2000000000",
            CUNUMERIC_MIN_GPU_CHUNK="2000000000",
        )

        result = system.run(cmd, env=env)
        self._log_proc(result, test_file, config.verbose)
        return result


STAGES: tuple[Type[TestStage], ...] = (CPU, GPU, OMP, Eager)
