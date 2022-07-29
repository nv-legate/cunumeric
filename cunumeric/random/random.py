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
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np
from cunumeric.array import ndarray
from cunumeric.runtime import runtime

from cunumeric.random import generator

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..types import NdShapeLike


def seed(init: Union[int, None] = None) -> None:
    if init is None:
        init = 0
    runtime.set_next_random_epoch(int(init))


###


def beta(
    a: float,
    b: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().beta(a, b, size, dtype)


def binomial(
    ntrials: int,
    p: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.uint32,
) -> ndarray:
    return generator.get_static_generator().binomial(ntrials, p, size, dtype)


def bytes(length: int) -> ndarray:
    return generator.get_static_generator().bytes(length)


def chisquare(
    df: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().chisquare(df, size, dtype)


def exponential(
    scale: float = 1.0,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().exponential(scale, size, dtype)


def f(
    dfnum: float,
    dfden: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().f(dfnum, dfden, size, dtype)


def gamma(
    shape: float,
    scale: float = 1.0,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().gamma(shape, scale, size, dtype)


def geometric(
    p: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.uint32,
) -> ndarray:
    return generator.get_static_generator().geometric(p, size, dtype)


def gumbel(
    loc: float = 0.0,
    scale: float = 1.0,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().gumbel(loc, scale, size, dtype)


def hypergeometric(
    ngood: int,
    nbad: int,
    nsample: int,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.uint32,
) -> ndarray:
    return generator.get_static_generator().hypergeometric(
        ngood, nbad, nsample, size, dtype
    )


def laplace(
    loc: float = 0.0,
    scale: float = 1.0,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().laplace(loc, scale, size, dtype)


def logistic(
    loc: float = 0.0,
    scale: float = 1.0,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().logistic(loc, scale, size, dtype)


def lognormal(
    mean: float = 0.0,
    sigma: float = 1.0,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().lognormal(mean, sigma, size, dtype)


def logseries(
    p: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.uint32,
) -> ndarray:
    return generator.get_static_generator().logseries(p, size, dtype)


def negative_binomial(
    ntrials: int,
    p: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.uint32,
) -> ndarray:
    return generator.get_static_generator().negative_binomial(
        ntrials, p, size, dtype
    )


def noncentral_chisquare(
    df: float,
    nonc: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().noncentral_chisquare(
        df, nonc, size, dtype
    )


def noncentral_f(
    dfnum: float,
    dfden: float,
    nonc: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().noncentral_f(
        dfnum, dfden, nonc, size, dtype
    )


def normal(
    loc: float = 0.0,
    scale: float = 1.0,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().normal(loc, scale, size, dtype)


def pareto(
    a: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().pareto(a, size, dtype)


def poisson(
    lam: float = 1.0, size: Union[NdShapeLike, None] = None
) -> ndarray:
    return generator.get_static_generator().poisson(lam, size)


def power(
    a: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().power(a, size, dtype)


def rand(*shapeargs: int) -> Union[float, ndarray]:
    return uniform(0.0, 1.0, size=shapeargs, dtype=np.float64)


def randint(
    low: int,
    high: Union[int, None] = None,
    size: Union[NdShapeLike, None] = None,
    dtype: Union[np.dtype[Any], type, None] = int,
) -> Union[int, ndarray, npt.NDArray[Any]]:
    return generator.get_static_generator().integers(low, high, size, dtype)


def randn(*shapeargs: int) -> Union[float, ndarray]:
    return normal(0.0, 1.0, shapeargs)


def random(
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
    out: Union[ndarray, None] = None,
) -> Union[float, ndarray]:
    return generator.get_static_generator().random(size, dtype, out)


# deprecated in numpy from version 1.11.0
def random_integers(
    low: int,
    high: Union[int, None] = None,
    size: Union[NdShapeLike, None] = None,
    dtype: Union[np.dtype[Any], type, None] = int,
) -> Union[int, ndarray, npt.NDArray[Any]]:
    if high is None:
        high = low
        low = 0
    return randint(low, high + 1, size, dtype)


def random_sample(
    size: Union[NdShapeLike, None] = None, dtype: npt.DTypeLike = np.float64
) -> Union[float, ndarray]:
    return uniform(0.0, 1.0, size, dtype)


ranf = random_sample


def rayleigh(
    scale: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().rayleigh(scale, size, dtype)


sample = random_sample


def standard_cauchy(
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().standard_cauchy(size, dtype)


def standard_exponential(
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().standard_exponential(size, dtype)


def standard_gamma(
    shape: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().standard_gamma(shape, size, dtype)


def standard_t(
    df: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().standard_t(df, size, dtype)


def triangular(
    left: float,
    mode: float,
    right: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().triangular(
        left, mode, right, size, dtype
    )


def uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().uniform(low, high, size, dtype)


def vonmises(
    mu: float,
    kappa: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().vonmises(mu, kappa, size, dtype)


def wald(
    mean: float,
    scale: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().wald(mean, scale, size, dtype)


def weibull(
    a: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.float64,
) -> ndarray:
    return generator.get_static_generator().weibull(a, size, dtype)


def zipf(
    a: float,
    size: Union[NdShapeLike, None] = None,
    dtype: npt.DTypeLike = np.uint32,
) -> ndarray:
    return generator.get_static_generator().zipf(a, size, dtype)
