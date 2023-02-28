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
import sys

import numpy as np
import pytest
from utils.random import ModuleGenerator, assert_distribution

import cunumeric as num

if sys.platform == "darwin":
    pytestmark = pytest.mark.skip()
    BITGENERATOR_ARGS = []
else:
    BITGENERATOR_ARGS = [
        ModuleGenerator,
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
    t(42, True)


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


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_integers_int16(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.integers(512, 653548, size=(1024,), dtype=np.int16)
    print(f"1024 sum = {a.sum()}")
    a = gen.integers(512, 653548, size=(1024 * 1024,), dtype=np.int16)
    print(f"1024*1024 sum = {a.sum()}")


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_random_float32(t):
    # top-level random function has a different signature
    if t is ModuleGenerator:
        return
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.random(size=(1024 * 1024,), dtype=np.float32)
    assert_distribution(a, 0.5, num.sqrt(1.0 / 12.0))


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_random_float64(t):
    # top-level random function has a different signature
    if t == ModuleGenerator:
        return
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    a = gen.random(size=(1024 * 1024,), dtype=np.float64)
    assert_distribution(a, 0.5, num.sqrt(1.0 / 12.0))


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_lognormal_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    sigma = 0.7
    a = gen.lognormal(mu, sigma, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = num.exp(mu + sigma * sigma / 2.0)
    theo_std = num.sqrt(
        (num.exp(sigma * sigma) - 1) * num.exp(2 * mu + sigma * sigma)
    )
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_lognormal_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    sigma = 0.7
    a = gen.lognormal(mu, sigma, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = num.exp(mu + sigma * sigma / 2.0)
    theo_std = num.sqrt(
        (num.exp(sigma * sigma) - 1) * num.exp(2 * mu + sigma * sigma)
    )
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_normal_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    sigma = 0.7
    a = gen.normal(mu, sigma, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = mu
    theo_std = sigma
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_normal_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    sigma = 0.7
    a = gen.normal(mu, sigma, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = mu
    theo_std = sigma
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_uniform_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    low = 1.414
    high = 3.14
    a = gen.uniform(low, high, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = (low + high) / 2.0
    theo_std = (high - low) / np.sqrt(12.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_uniform_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    low = 1.414
    high = 3.14
    a = gen.uniform(low, high, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = (low + high) / 2.0
    theo_std = (high - low) / np.sqrt(12.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_poisson(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    lam = 42
    a = gen.poisson(lam, size=(1024 * 1024,))
    theo_mean = lam
    theo_std = num.sqrt(lam)
    assert_distribution(a, theo_mean, theo_std)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
