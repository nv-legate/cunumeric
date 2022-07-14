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

from .. import FeatureType
from ..config import Config
from ..system import System
from ..types import ArgList, EnvDict
from .test_stage import Shard, StageSpec, TestStage, adjust_workers


class CPU(TestStage):
    """A test stage for exercising CPU features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: System
        Process execution wrapper

    """

    kind: FeatureType = "cpus"

    args = ["-cunumeric:test"]

    env: EnvDict = {}

    def __init__(self, config: Config, system: System) -> None:
        self._init(config, system)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        return [
            "--cpus",
            str(len(shard)),
            "--cpu-bind",
            ",".join(str(x) for x in shard),
        ]

    def compute_spec(self, config: Config, system: System) -> StageSpec:
        if config.verbose:
            # use all available CPUs for a single worker
            return StageSpec(1, [tuple(range(len(system.cpus)))])

        N = len(system.cpus)

        degree = N // (config.cpus + config.utility)
        workers = adjust_workers(degree, config.requested_workers)

        # https://docs.python.org/3/library/itertools.html#itertools-recipes
        # grouper('ABCDEF', 3) --> ABC DEF
        args = [iter(range(workers * config.cpus))] * config.cpus
        shards = list(zip(*args))

        return StageSpec(workers, shards)
