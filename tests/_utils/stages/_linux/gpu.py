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

from ... import FeatureType
from ...config import Config
from ...system import System
from ...types import ArgList, EnvDict
from ..test_stage import TestStage
from ..util import CUNUMERIC_TEST_ARG, Shard, StageSpec, adjust_workers

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

    args = [CUNUMERIC_TEST_ARG]

    def __init__(self, config: Config, system: System) -> None:
        self._init(config, system)

    def env(self, config: Config, system: System) -> EnvDict:
        return {}

    def shard_args(self, shard: Shard, config: Config) -> ArgList:
        return [
            "--fbmem",
            str(config.fbmem),
            "--gpus",
            str(len(shard)),
            "--gpu-bind",
            ",".join(str(x) for x in shard),
        ]

    def compute_spec(self, config: Config, system: System) -> StageSpec:
        N = len(system.gpus)
        degree = N // config.gpus

        fbsize = min(gpu.total for gpu in system.gpus) / (2 << 20)  # MB
        oversub_factor = int(fbsize // (config.fbmem * BLOAT_FACTOR))
        workers = adjust_workers(
            degree * oversub_factor, config.requested_workers
        )

        # https://docs.python.org/3/library/itertools.html#itertools-recipes
        # grouper('ABCDEF', 3) --> ABC DEF
        args = [iter(range(degree * config.gpus))] * config.gpus
        per_worker_shards = list(zip(*args))

        shards = per_worker_shards * workers

        return StageSpec(workers, shards)
