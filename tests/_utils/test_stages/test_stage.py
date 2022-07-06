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
from typing import Tuple

from typing_extensions import Protocol, TypeAlias

from .. import PER_FILE_ARGS, FeatureType
from ..config import Config
from ..logger import LOG
from ..system import SKIPPED_RETURNCODE, System
from ..types import ArgList, EnvDict
from ..ui import banner, failed, passed, skipped, summary, yellow

Shard: TypeAlias = Tuple[int, ...]


@dataclass(frozen=True)
class StageSpec:
    """Specify the operation of a test run"""

    #: The number of worker processes to start for running tests
    workers: int

    # A list of (cpu or gpu) shards to draw on for each test
    shards: list[Shard]


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

    kind: FeatureType

    #: The computed specification for processes to launch to run the
    #: configured test files.
    spec: StageSpec

    #: The computed sharding id sets to use for job runs
    shards: multiprocessing.Queue[Shard]

    #: After the stage completes, results will be stored here
    result: StageResult

    #: Any fixed stage-specific command-line args to pass
    args: ArgList = []

    #: Any stage-specific customizations to the process env
    env: EnvDict = {}

    _manager: multiprocessing.managers.SyncManager = multiprocessing.Manager()

    # --- Protocol methods

    def __init__(self, config: Config, system: System) -> None:
        ...

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        """Generate the command line arguments necessary to launch
        the next test process on the given shard.

        """
        ...

    def compute_spec(self, config: Config, system: System) -> StageSpec:
        """Compute the number of worker processes to launch and stage shards
        to use for running the configured test files.

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
        workers = self.spec.workers
        workers_text = f"{workers} worker{'s' if workers > 1 else ''}"
        return (
            banner(f"Entering stage: {self.name} (with {workers_text})") + "\n"
        )

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

        # These are a bit ugly but necessary in order to make pytest generate
        # more verbose output for integration tests when -v, -vv is specified
        if "integration" in test_file_string and config.verbose > 0:
            args += ["-v"]
        if "integration" in test_file_string and config.verbose > 1:
            args += ["-s"]

        return args

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
        test_path = config.root_dir / test_file

        shard = self.shards.get()

        stage_args = self.args + self.shard_args(shard, config)
        file_args = self.file_args(test_file, config)

        cmd = [str(config.legate_path), str(test_path)]
        cmd += stage_args + file_args + config.extra_args

        result = system.run(cmd, env=self._env(config))
        self._log_proc(result, test_file, config.verbose)

        self.shards.put(shard)

        return result

    def _adjust_workers(self, workers: int, config: Config) -> int:
        if config.requested_workers is not None:
            if config.requested_workers > workers:
                raise RuntimeError(
                    "Requested workers greater than assignable workers"
                )
            workers = config.requested_workers

        if workers == 0:
            raise RuntimeError("Current configuration results in zero workers")

        return workers

    def _env(self, config: Config) -> EnvDict:
        env = config.env
        env.update(self.env)
        return env

    def _init(self, config: Config, system: System) -> None:
        self.spec = self.compute_spec(config, system)
        self.shards = self._manager.Queue(len(self.spec.shards))
        for shard in self.spec.shards:
            self.shards.put(shard)

    def _launch(
        self, config: Config, system: System
    ) -> list[CompletedProcess[str]]:

        pool = multiprocessing.Pool(self.spec.workers)

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
