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

from . import DEFAULT_GPU_MEMORY_BUDGET, DEFAULT_GPU_PARALLELISM, PER_FILE_ARGS
from .config import Config
from .logger import LOG
from .system import SKIPPED_RETURNCODE, ArgList, System
from .ui import banner, failed, passed, skipped, summary, yellow


@dataclass(frozen=True)
class StageResult:
    """Collect results from all tests in a TestStage."""

    #: Individual test process results including return code and stdout.
    procs: list[CompletedProcess[str]]

    #: Cumulative execution time for all tests in a stage.
    time: timedelta


class TestStage(Protocol):
    """Encapsulate running configured test files using specific features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: System
        Process execution wrapper

    """

    kind: str

    #: The computed number of worker processes to launch to run the
    #: configured test files.
    workers: int

    #: After the stage completes, results will be stored here
    result: StageResult

    # --- Protocol methods

    def __init__(self, config: Config, system: System) -> None:
        ...

    def run(
        self, test_file: Path, config: Config, system: System
    ) -> CompletedProcess[str]:
        """Execute a single test files with appropriate environment and
        command-line options for a feature test stage.

        Parameters
        ----------
        test_file : Path
            Test file to execute

        config: Config
            Test runner configuration

        system: System
            Process execution wrapper

        """
        ...

    def compute_workers(self, config: Config, system: System) -> int:
        """Compute the number of worker processes to launch for running
        the configured test files.

        Parameters
        ----------
        config: Config
            Test runner configuration

        system: System
            Process execution wrapper

        """
        ...

    # --- Shared implementation methods

    def __call__(self, config: Config, system: System) -> None:
        """Execute this test stage.

        Parameters
        ----------
        config: Config
            Test runner configuration

        system: System
            Process execution wrapper

        """
        t0 = datetime.now()
        procs = self._launch(config, system)
        t1 = datetime.now()

        self.result = StageResult(procs, t1 - t0)

    @property
    def name(self) -> str:
        """A stage name to display for tests in this stage."""
        return self.__class__.__name__

    @property
    def intro(self) -> str:
        """An informative banner to display at stage end."""
        workers = f"{self.workers} worker{'s' if self.workers > 1 else ''}"
        return banner(f"Entering stage: {self.name} (with {workers})") + "\n"

    @property
    def outro(self) -> str:
        """An informative banner to display at stage end."""
        total = len(self.result.procs)
        passed = len([p for p in self.result.procs if p.returncode == 0])

        result = summary(self.name, total, passed)

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

        return f"{result}\n{footer}"

    def file_args(self, test_file: Path, config: Config) -> ArgList:
        """Extra command line arguments based on the test file.

        Parameters
        ----------
        test_file : Path
            Path to a test file

        config: Config
            Test runner configuration

        """
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

        pool = multiprocessing.Pool(self.workers)

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

        cmd = [str(config.legate_path), str(test_path)]
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


class OMP(TestStage):
    """A test stage for exercising OpenMP features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: System
        Process execution wrapper

    """

    kind = "openmp"

    def __init__(self, config: Config, system: System) -> None:
        self.workers = self.compute_workers(config, system)

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

    def omp_args(self, config: Config) -> ArgList:
        return [
            "--omps",
            str(config.omps),
            "--ompthreads",
            str(config.ompthreads),
        ]

    def compute_workers(self, config: Config, system: System) -> int:
        if config.requested_workers is not None:
            return config.requested_workers

        requested_procs = config.omps * config.ompthreads
        return 1 if config.verbose else len(system.cpus) // requested_procs


class Eager(TestStage):
    """A test stage for exercising Eager Numpy execution features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: System
        Process execution wrapper

    """

    kind = "eager"

    def __init__(self, config: Config, system: System) -> None:
        self.workers = self.compute_workers(config, system)

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

    def compute_workers(self, config: Config, system: System) -> int:
        if config.requested_workers is not None:
            return config.requested_workers

        return 1 if config.verbose else len(system.cpus)


#: All the available test stages that can be selected
STAGES: tuple[Type[TestStage], ...] = (CPU, GPU, OMP, Eager)
