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

from typing import TYPE_CHECKING, Union

import numpy as np

from ..array import ndarray
from ..config import BitGeneratorType
from ..runtime import runtime

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..types import NdShapeLike


class BitGenerator:
    def __init__(
        self,
        seed: Union[int, None] = None,
        generatorType: BitGeneratorType = BitGeneratorType.DEFAULT,
        forceBuild: bool = False,
    ) -> None:
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

    def __del__(self) -> None:
        if self.handle != 0:
            runtime.bitgenerator_destroy(self.handle, disposing=True)

    # when output is false => skip ahead
    def random_raw(self, shape: Union[NdShapeLike, None] = None) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(np.uint32))
        res._thunk.bitgenerator_random_raw(
            self.handle, self.generatorType, self.seed, self.flags
        )
        return res

    def integers(
        self,
        low: int,
        high: Union[int, None] = None,
        shape: Union[NdShapeLike, None] = None,
        type: npt.DTypeLike = np.int64,
        endpoint: bool = False,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(type))
        if high is None:
            high = low
            low = 0
        if endpoint:
            high = high + 1
        res._thunk.bitgenerator_integers(
            self.handle, self.generatorType, self.seed, self.flags, low, high
        )
        return res

    def random(
        self,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
        res: Union[ndarray, None] = None,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        if res is None:
            res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_uniform(
            self.handle, self.generatorType, self.seed, self.flags, 0, 1
        )
        return res

    def lognormal(
        self,
        mean: float = 0.0,
        sigma: float = 1.0,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_lognormal(
            self.handle, self.generatorType, self.seed, self.flags, mean, sigma
        )
        return res

    def normal(
        self,
        mean: float = 0.0,
        sigma: float = 1.0,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_normal(
            self.handle, self.generatorType, self.seed, self.flags, mean, sigma
        )
        return res

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_uniform(
            self.handle, self.generatorType, self.seed, self.flags, low, high
        )
        return res

    def poisson(
        self, lam: float, shape: Union[NdShapeLike, None] = None
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(np.uint32))
        res._thunk.bitgenerator_poisson(
            self.handle, self.generatorType, self.seed, self.flags, lam
        )
        return res

    def exponential(
        self,
        scale: float = 1.0,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_exponential(
            self.handle, self.generatorType, self.seed, self.flags, scale
        )
        return res

    def gumbel(
        self,
        mu: float = 0.0,
        beta: float = 1.0,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_gumbel(
            self.handle, self.generatorType, self.seed, self.flags, mu, beta
        )
        return res

    def laplace(
        self,
        mu: float = 0.0,
        beta: float = 1.0,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_laplace(
            self.handle, self.generatorType, self.seed, self.flags, mu, beta
        )
        return res

    def logistic(
        self,
        mu: float = 0.0,
        beta: float = 1.0,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_logistic(
            self.handle, self.generatorType, self.seed, self.flags, mu, beta
        )
        return res

    def pareto(
        self,
        alpha: float,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_pareto(
            self.handle, self.generatorType, self.seed, self.flags, alpha
        )
        return res

    def power(
        self,
        alpha: float,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_power(
            self.handle, self.generatorType, self.seed, self.flags, alpha
        )
        return res

    def rayleigh(
        self,
        sigma: float,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_rayleigh(
            self.handle, self.generatorType, self.seed, self.flags, sigma
        )
        return res

    def cauchy(
        self,
        x0: float,
        gamma: float,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_cauchy(
            self.handle, self.generatorType, self.seed, self.flags, x0, gamma
        )
        return res

    def triangular(
        self,
        a: float,
        b: float,
        c: float,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_triangular(
            self.handle, self.generatorType, self.seed, self.flags, a, b, c
        )
        return res

    def weibull(
        self,
        lam: float,
        k: float,
        shape: Union[NdShapeLike, None] = None,
        dtype: npt.DTypeLike = np.float64,
    ) -> ndarray:
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_weibull(
            self.handle, self.generatorType, self.seed, self.flags, lam, k
        )
        return res

    def bytes(self, length: Union[int, tuple[int, ...]]) -> ndarray:
        if not isinstance(length, tuple):
            length = (length,)
        res = ndarray(length, dtype=np.dtype(np.uint8))
        res._thunk.bitgenerator_bytes(
            self.handle,
            self.generatorType,
            self.seed,
            self.flags,
        )
        return res


class XORWOW(BitGenerator):
    def __init__(
        self, seed: Union[int, None] = None, forceBuild: bool = False
    ) -> None:
        super().__init__(seed, BitGeneratorType.XORWOW, forceBuild)


class MRG32k3a(BitGenerator):
    def __init__(
        self, seed: Union[int, None] = None, forceBuild: bool = False
    ) -> None:
        super().__init__(seed, BitGeneratorType.MRG32K3A, forceBuild)


class PHILOX4_32_10(BitGenerator):
    def __init__(
        self, seed: Union[int, None] = None, forceBuild: bool = False
    ) -> None:
        super().__init__(seed, BitGeneratorType.PHILOX4_32_10, forceBuild)
