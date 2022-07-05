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
    print(f"average = {average}, stdev = {stdev}\n")
    assert np.abs(theo_mean - average) < tolerance * np.max(
        (1.0, np.abs(theo_mean))
    )
    assert np.abs(theo_stdev - stdev) < tolerance * np.max(
        (1.0, np.abs(theo_stdev))
    )


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_exponential_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    a = gen.exponential(scale, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = scale
    theo_std = scale
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_exponential_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    a = gen.exponential(scale, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = scale
    theo_std = scale
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_gumbel_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.gumbel(loc, scale, size=(1024 * 1024,), dtype=np.float32)
    euler_mascheroni = 0.5772156649015328606065120900824024310421
    theo_mean = loc + euler_mascheroni * scale
    theo_std = np.pi * scale / np.sqrt(6.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_gumbel_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.gumbel(loc, scale, size=(1024 * 1024,), dtype=np.float64)
    euler_mascheroni = 0.5772156649015328606065120900824024310421
    theo_mean = loc + euler_mascheroni * scale
    theo_std = np.pi * scale / np.sqrt(6.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_laplace_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.laplace(loc, scale, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = loc
    theo_std = np.sqrt(2.0) * scale
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_laplace_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.laplace(loc, scale, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = loc
    theo_std = np.sqrt(2.0) * scale
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_logistic_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.logistic(loc, scale, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = loc
    theo_std = np.pi * scale / np.sqrt(3.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_logistic_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    scale = 1.414
    loc = 0.7
    a = gen.logistic(loc, scale, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = loc
    theo_std = np.pi * scale / np.sqrt(3.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_pareto_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    alpha = 30.0
    a = gen.pareto(alpha, size=(1024 * 1024,), dtype=np.float32)
    xm = 1.0
    theo_mean = alpha * xm / (alpha - 1.0) - 1.0
    theo_std = np.sqrt(xm * xm * alpha / (alpha - 2)) / (alpha - 1)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_pareto_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    alpha = 30.0
    a = gen.pareto(alpha, size=(1024 * 1024,), dtype=np.float64)
    xm = 1.0
    theo_mean = alpha * xm / (alpha - 1.0) - 1.0
    theo_std = np.sqrt(xm * xm * alpha / (alpha - 2)) / (alpha - 1)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_power_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    alpha = 3.0
    a = gen.power(alpha, size=(1024 * 1024,), dtype=np.float32)
    # power function distribution is Beta distribution with
    # alpha > 0 and beta = 1
    beta = 1.0
    theo_mean = alpha / (alpha + beta)
    theo_std = np.sqrt((alpha * beta) / (alpha + beta + 1.0)) / (alpha + beta)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_power_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    alpha = 3.0
    a = gen.power(alpha, size=(1024 * 1024,), dtype=np.float64)
    # power function distribution is Beta distribution with
    # alpha > 0 and beta = 1
    beta = 1.0
    theo_mean = alpha / (alpha + beta)
    theo_std = np.sqrt((alpha * beta) / (alpha + beta + 1.0)) / (alpha + beta)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_rayleigh_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    sigma = np.pi
    a = gen.rayleigh(sigma, size=(1024 * 1024,), dtype=np.float32)
    theo_mean = sigma * np.sqrt(np.pi / 2.0)
    theo_std = sigma * np.sqrt((4.0 - np.pi) / 2.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_rayleigh_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    sigma = np.pi
    a = gen.rayleigh(sigma, size=(1024 * 1024,), dtype=np.float64)
    theo_mean = sigma * np.sqrt(np.pi / 2.0)
    theo_std = sigma * np.sqrt((4.0 - np.pi) / 2.0)
    assert_distribution(a, theo_mean, theo_std)


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_cauchy_float32(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    x0 = 1.414
    a = gen.cauchy(x0, 1.0, size=(1024 * 1024,), dtype=np.float32)
    # cauchy's distribution mean and stdev are not defined...
    aa = np.array(a)
    median = np.median(aa)
    assert np.abs(median - x0) < 0.01


@pytest.mark.parametrize("t", BITGENERATOR_ARGS, ids=str)
def test_cauchy_float64(t):
    bitgen = t(seed=42)
    gen = num.random.Generator(bitgen)
    x0 = 1.414
    a = gen.cauchy(x0, 1.0, size=(1024 * 1024,), dtype=np.float64)
    # cauchy's distribution mean and stdev are not defined...
    aa = np.array(a)
    median = np.median(aa)
    assert np.abs(median - x0) < 0.01


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
