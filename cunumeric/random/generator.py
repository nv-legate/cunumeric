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
from cunumeric.random.bitgenerator import XORWOW, BitGenerator


class Generator:
    def __init__(self, bit_generator):
        self.bit_generator = bit_generator

    def integers(
        self, low, high=None, size=None, dtype=np.int64, endpoint=False
    ):
        return self.bit_generator.integers(low, high, size, dtype, endpoint)


def default_rng(seed=None):
    if seed is None:
        return Generator(XORWOW())
    elif isinstance(seed, BitGenerator):
        return Generator(seed)
    elif isinstance(seed, Generator):
        return seed
    else:
        return Generator(XORWOW(seed))
