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

from typing import Any

from ...system import System
from ...types import CPUInfo, GPUInfo


class FakeSystem(System):
    def __init__(
        self, cpus: int = 6, gpus: int = 6, fbmem: int = 6 << 32, **kwargs: Any
    ) -> None:
        self._cpus = cpus
        self._gpus = gpus
        self._fbmem = fbmem
        super().__init__(**kwargs)

    @property
    def cpus(self) -> tuple[CPUInfo, ...]:
        return tuple(CPUInfo((i,)) for i in range(self._cpus))

    @property
    def gpus(self) -> tuple[GPUInfo, ...]:
        return tuple(GPUInfo(i, self._fbmem) for i in range(self._gpus))
