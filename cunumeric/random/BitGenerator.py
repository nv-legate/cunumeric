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
from cunumeric.config import CuNumericOpCode
from cunumeric.runtime import runtime

class BitGenerator:
    # see bitgenerator_util.h
    OP_CREATE = 1
    OP_DESTROY = 2
    OP_RAND_RAW = 3

    # see bitgenerator_util.h
    DEFAULT = 0
    XORWOW = 1
    MRG32K3A = 2
    MTGP32 = 3
    MT19937 = 4
    PHILOX4_32_10 = 5
    
    __slots__ = [
        "handle", # handle to the runtime id
    ]

    def __init__(self, seed=None, generatorType=DEFAULT):
        self.handle = runtime.bitgenerator_create(generatorType)

    def __del__(self):
        runtime.bitgenerator_destroy(self.handle)

    # when output is false => skip ahead
    def random_raw(self, shape=None, output=True):
        if shape is None:
            raise NotImplementedError('Empty shape not implemented')
        if not isinstance(shape, tuple):
            shape = (shape,)
        if output:
            res = ndarray(shape,dtype=np.dtype(np.uint32))
            res.bitgenerator_random_raw(self.handle)
            return res
        else:
            return runtime.bitgenerator_random_raw(self.handle, shape)

class XORWOW(BitGenerator):
    def __init__(self, seed=None):
        super().__init__(seed, BitGenerator.XORWOW)

class MRG32k3a(BitGenerator):
    def __init__(self, seed=None):
        super().__init__(seed, BitGenerator.MRG32K3A)

class MTGP32(BitGenerator):
    def __init__(self, seed=None):
        super().__init__(seed, BitGenerator.MTGP32)

class MT19937(BitGenerator):
    def __init__(self, seed=None):
        super().__init__(seed, BitGenerator.MT19937)

class PHILOX4_32_10(BitGenerator):
    def __init__(self, seed=None):
        super().__init__(seed, BitGenerator.PHILOX4_32_10)

