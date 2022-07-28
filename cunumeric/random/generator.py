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

from typing import TYPE_CHECKING, Union

import numpy as np
from cunumeric.random.bitgenerator import XORWOW, BitGenerator

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..array import ndarray
    from ..types import NdShapeLike


class Generator:
    def __init__(self, bit_generator: BitGenerator) -> None:
        self.bit_generator = bit_generator

    def integers(
        self,
        low: int,
        high: Union[int, None] = None,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.int64,
        endpoint: bool = False,
    ) -> ndarray:
        return self.bit_generator.integers(low, high, size, dtype, endpoint)

    def random(
        self,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
        out: Union[ndarray, None] = None,
    ) -> ndarray:
        if out is not None:
            if size is not None:
                assert out.shape == size
            assert out.dtype == dtype
        return self.bit_generator.random(size, dtype, out)

    def lognormal(
        self,
        mean: float = 0.0,
        sigma: float = 1.0,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.lognormal(mean, sigma, size, dtype)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.normal(
            mean=loc, sigma=scale, shape=size, dtype=dtype
        )

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.uniform(low, high, size, dtype)

    def poisson(
        self, lam: float = 1.0, size: Union[NdShapeLike, None] = None
    ) -> ndarray:
        return self.bit_generator.poisson(lam, size)

    def exponential(
        self,
        scale: float = 1.0,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.exponential(
            scale=scale, shape=size, dtype=dtype
        )

    def gumbel(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.gumbel(
            mu=loc, beta=scale, shape=size, dtype=dtype
        )

    def laplace(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.laplace(
            mu=loc, beta=scale, shape=size, dtype=dtype
        )

    def logistic(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.logistic(
            mu=loc, beta=scale, shape=size, dtype=dtype
        )

    def pareto(
        self,
        a: float,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.pareto(alpha=a, shape=size, dtype=dtype)

    def power(
        self,
        a: float,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.power(alpha=a, shape=size, dtype=dtype)

    def rayleigh(
        self,
        scale: float,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.rayleigh(
            sigma=scale, shape=size, dtype=dtype
        )

    def cauchy(
        self,
        x0: float,
        gamma: float,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.cauchy(
            x0=x0, gamma=gamma, shape=size, dtype=dtype
        )

    def standard_cauchy(
        self,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.cauchy(0.0, 1.0, size, dtype)

    def standard_exponential(
        self,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.exponential(1.0, size, dtype)

    def triangular(
        self,
        left: float,
        mode: float,
        right: float,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.triangular(
            a=left, b=right, c=mode, shape=size, dtype=dtype
        )

    def weibull(
        self,
        a: float,
        size: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        return self.bit_generator.weibull(lam=1, k=a, shape=size, dtype=dtype)

    def bytes(self, length: Union[int, tuple[int, ...]]) -> ndarray:
        return self.bit_generator.bytes(length=length)


def default_rng(seed: Union[int, None] = None) -> Generator:
    """Construct a new Generator with the default BitGenerator (PCG64).

    Parameters
    ----------
    seed :
        {None, int, array_like[ints], SeedSequence, BitGenerator, Generator},
        optional

        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.
        Additionally, when passed a `BitGenerator`, it will be wrapped by
        `Generator`. If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    Generator
        The initialized generator object.

    Notes
    -----
    If ``seed`` is not a `BitGenerator` or a `Generator`, a new `BitGenerator`
    is instantiated. This function does not manage a default global instance.

    """
    if seed is None:
        return Generator(XORWOW())
    elif isinstance(seed, BitGenerator):
        return Generator(seed)
    elif isinstance(seed, Generator):
        return seed
    else:
        return Generator(XORWOW(seed))
