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
import pytest

import cunumeric as num

BITGENERATOR_ARGS = [
    num.random.XORWOW,
    num.random.MRG32k3a,
    num.random.PHILOX4_32_10,
]


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_bitgenerator_type(t):
    print(f"testing for type = {t}")
    bitgen = t(seed=42)
    bitgen.random_raw(256)
    bitgen.random_raw((512, 256))
    r = bitgen.random_raw(256)  # deferred is None
    print(f"256 sum = {r.sum()}")
    r = bitgen.random_raw((1024, 1024))
    print(f"1k² sum = {r.sum()}")
    r = bitgen.random_raw(1024 * 1024)
    print(f"1M sum = {r.sum()}")
    bitgen = None
    print(f"DONE for type = {t}")


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_force_build(t):
    bitgen = t(42, True)
    bitgen.destroy()


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_integers_int64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.integers(512, 653548, size=(1024,))
    print(f"1024 sum = {a.sum()}")
    a = gen.integers(512, 653548, size=(1024 * 1024,))
    print(f"1024*1024 sum = {a.sum()}")


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_integers_int32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.integers(512, 653548, size=(1024,), dtype=np.int32)
    print(f"1024 sum = {a.sum()}")
    a = gen.integers(512, 653548, size=(1024 * 1024,), dtype=np.int32)
    print(f"1024*1024 sum = {a.sum()}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
