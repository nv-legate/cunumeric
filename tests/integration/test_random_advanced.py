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
import numpy as np
import pytest

import cunumeric as num

BITGENERATOR_ARGS = [
    num.random.XORWOW,
    num.random.MRG32k3a,
    num.random.PHILOX4_32_10,
]


def assert_distribution(a, theo_mean, theo_stdev, tolerance=1e-2):
    if True:
        aa = np.array(a)
        average = np.mean(aa)
        stdev = np.std(aa)
    else:  # keeping this path for further investigation
        average = num.mean(a)
        stdev = num.sqrt(
            num.mean((a - average) ** 2)
        )  # num.std(a) -> does not work
    print(
        f"average = {average} - expected {theo_mean}"
        + f", stdev = {stdev} - expected {theo_stdev}\n"
    )
    assert np.abs(theo_mean - average) < tolerance * np.max(
        (1.0, np.abs(theo_mean))
    )
    assert np.abs(theo_stdev - stdev) < tolerance * np.max(
        (1.0, np.abs(theo_stdev))
    )


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_standard_t_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    nu = 3.1415
    a = gen.standard_t(df=nu, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = 0
    theo_std = np.sqrt(nu / (nu - 2.0))
    assert_distribution(a, theo_mean, theo_std, 0.1)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_standard_t_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    nu = 3.1415
    a = gen.standard_t(df=nu, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = 0
    theo_std = np.sqrt(nu / (nu - 2.0))
    assert_distribution(a, theo_mean, theo_std, 0.1)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_vonmises_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    kappa = 3.1415
    a = gen.vonmises(mu, kappa, size=(1024 * 1024,), dtype=np.float32)
    ref_a = np.random.vonmises(mu, kappa, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_vonmises_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    kappa = 3.1415
    a = gen.vonmises(mu, kappa, size=(1024 * 1024,), dtype=np.float64)
    ref_a = np.random.vonmises(mu, kappa, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_hypergeometric(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    N = 500
    K = 60
    n = 200
    ngood = K
    nbad = N - K
    nsample = n
    a = gen.hypergeometric(
        ngood, nbad, nsample, size=(1024 * 1024,), dtype=np.uint32
    )
    theo_mean = n * K / N
    theo_std = np.sqrt(n * (K * (N - K) * (N - n)) / (N * N * (N - 1)))
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_geometric(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    p = 0.707
    a = gen.geometric(p, size=(1024 * 1024,), dtype=np.uint32)
    theo_mean = 1 / p
    theo_std = np.sqrt(1 - p) / p
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_zipf(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    s = 7.5
    a = gen.zipf(a=s, size=(1024 * 1024,), dtype=np.uint32)
    a = np.random.zipf(s, 1024 * 1024)
    ref_a = np.random.zipf(s, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std, 0.2)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_wald_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    kappa = 3.1415
    a = gen.wald(mu, kappa, size=(1024 * 1024,), dtype=np.float32)
    ref_a = np.random.wald(mu, kappa, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_wald_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    mu = 1.414
    kappa = 3.1415
    a = gen.wald(mu, kappa, size=(1024 * 1024,), dtype=np.float64)
    ref_a = np.random.wald(mu, kappa, 1024 * 1024)
    theo_mean = np.average(ref_a)
    theo_std = np.std(ref_a)
    assert_distribution(a, theo_mean, theo_std)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
