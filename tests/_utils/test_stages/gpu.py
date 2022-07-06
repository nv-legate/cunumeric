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
from ..types import ArgList
from .test_stage import Shard, StageSpec, TestStage

BLOAT_FACTOR = 1.5  # hard coded for now


class GPU(TestStage):
    """A test stage for exercising GPU features.

    Parameters
    ----------
    config: Config
        Test runner configuration

    system: System
        Process execution wrapper

    """

    kind: FeatureType = "cuda"

    stage_args = ["-cunumeric:test"]

    def __init__(self, config: Config, system: System) -> None:
        self._init(config, system)

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        return [
            "--gpus",
            str(len(shard)),
            "--gpu-bind",
            ",".join(str(x) for x in shard),
        ]

    def compute_spec(self, config: Config, system: System) -> StageSpec:
        if config.verbose:
            # use all available GPUs for a single worker
            return StageSpec(1, [tuple(range(len(system.gpus)))])

        N = len(system.gpus)

        fbsize = min(gpu.total for gpu in system.gpus)
        oversub_factor = int(fbsize // (config.fbmem * BLOAT_FACTOR))

        degree = N // config.gpus
        workers = self._adjust_workers(degree * oversub_factor, config)

        # https://docs.python.org/3/library/itertools.html#itertools-recipes
        # grouper('ABCDEF', 3) --> ABC DEF
        args = [iter(range(degree * config.gpus))] * config.gpus
        per_worker_shards = list(zip(*args))

        shards = per_worker_shards * workers

        return StageSpec(workers, shards)
