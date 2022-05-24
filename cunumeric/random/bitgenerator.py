# Copyright 2021-2022 NVIDIA Corporation
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

import numpy as np
from cunumeric.array import ndarray
from cunumeric.config import BitGeneratorType
from cunumeric.runtime import runtime


class BitGenerator:
    def __init__(
        self,
        seed=None,
        generatorType=BitGeneratorType.DEFAULT,
        forceBuild=False,
    ):
        if type(self) is BitGenerator:
            raise NotImplementedError(
                "BitGenerator is a base class and cannot be instantized"
            )
        self.generatorType = generatorType
        self.seed = seed
        self.flags = 0
        self.handle = runtime.bitgenerator_create(
            generatorType, seed, self.flags, forceBuild
        )

    def __del__(self):
        if self.handle != 0:
            runtime.bitgenerator_destroy(self.handle, disposing=True)

    # explicit destruction
    def destroy(self):
        runtime.bitgenerator_destroy(self.handle, disposing=False)
        self.handle = 0

    # when output is false => skip ahead
    def random_raw(self, shape=None):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(np.uint32))
        res._thunk.bitgenerator_random_raw(
            self.handle, self.generatorType, self.seed, self.flags
        )
        return res


class XORWOW(BitGenerator):
    def __init__(self, seed=None, forceBuild=False):
        super().__init__(seed, BitGeneratorType.XORWOW, forceBuild)


class MRG32k3a(BitGenerator):
    def __init__(self, seed=None, forceBuild=False):
        super().__init__(seed, BitGeneratorType.MRG32K3A, forceBuild)


class PHILOX4_32_10(BitGenerator):
    def __init__(self, seed=None, forceBuild=False):
        super().__init__(seed, BitGeneratorType.PHILOX4_32_10, forceBuild)
