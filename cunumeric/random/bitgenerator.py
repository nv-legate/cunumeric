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

    def integers(
        self, low, high=None, shape=None, type=np.int64, endpoint=False
    ):
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

    def random(self, shape=None, dtype=np.float64, res=None):
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

    def lognormal(self, mean=0.0, sigma=1.0, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_lognormal(
            self.handle, self.generatorType, self.seed, self.flags, mean, sigma
        )
        return res

    def normal(self, mean=0.0, sigma=1.0, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_normal(
            self.handle, self.generatorType, self.seed, self.flags, mean, sigma
        )
        return res

    def uniform(self, low=0.0, high=1.0, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_uniform(
            self.handle, self.generatorType, self.seed, self.flags, low, high
        )
        return res

    def poisson(self, lam, shape=None):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(np.uint32))
        res._thunk.bitgenerator_poisson(
            self.handle, self.generatorType, self.seed, self.flags, lam
        )
        return res

    def exponential(self, scale=1.0, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_exponential(
            self.handle, self.generatorType, self.seed, self.flags, scale
        )
        return res

    def gumbel(self, mu=0.0, beta=1.0, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_gumbel(
            self.handle, self.generatorType, self.seed, self.flags, mu, beta
        )
        return res

    def laplace(self, mu=0.0, beta=1.0, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_laplace(
            self.handle, self.generatorType, self.seed, self.flags, mu, beta
        )
        return res

    def logistic(self, mu=0.0, beta=1.0, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_logistic(
            self.handle, self.generatorType, self.seed, self.flags, mu, beta
        )
        return res

    def pareto(self, alpha, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_pareto(
            self.handle, self.generatorType, self.seed, self.flags, alpha
        )
        return res

    def power(self, alpha, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_power(
            self.handle, self.generatorType, self.seed, self.flags, alpha
        )
        return res

    def rayleigh(self, sigma, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_rayleigh(
            self.handle, self.generatorType, self.seed, self.flags, sigma
        )
        return res

    def cauchy(self, x0, gamma, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_cauchy(
            self.handle, self.generatorType, self.seed, self.flags, x0, gamma
        )
        return res

    def triangular(self, a, b, c, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_triangular(
            self.handle, self.generatorType, self.seed, self.flags, a, b, c
        )
        return res

    def weibull(self, lam, k, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=np.dtype(dtype))
        res._thunk.bitgenerator_weibull(
            self.handle, self.generatorType, self.seed, self.flags, lam, k
        )
        return res

    def bytes(self, length):
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

    def beta(self, a, b, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_beta(
            self.handle, self.generatorType, self.seed, self.flags, a, b
        )
        return res

    def f(self, dfnum, dfden, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_f(
            self.handle,
            self.generatorType,
            self.seed,
            self.flags,
            dfnum,
            dfden,
        )
        return res

    def logseries(self, p, shape=None, dtype=np.uint32):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_logseries(
            self.handle, self.generatorType, self.seed, self.flags, p
        )
        return res

    def noncentral_f(self, dfnum, dfden, nonc, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_noncentral_f(
            self.handle,
            self.generatorType,
            self.seed,
            self.flags,
            dfnum,
            dfden,
            nonc,
        )
        return res

    def chisquare(self, df, nonc=0.0, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_chisquare(
            self.handle, self.generatorType, self.seed, self.flags, df, nonc
        )
        return res

    def gamma(self, k, theta=1.0, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_gamma(
            self.handle, self.generatorType, self.seed, self.flags, k, theta
        )
        return res

    def standard_t(self, df, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_standard_t(
            self.handle, self.generatorType, self.seed, self.flags, df
        )
        return res

    def hypergeometric(
        self, ngood, nbad, nsample, shape=None, dtype=np.uint32
    ):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_hypergeometric(
            self.handle,
            self.generatorType,
            self.seed,
            self.flags,
            ngood,
            nbad,
            nsample,
        )
        return res

    def vonmises(self, mu, kappa, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_vonmises(
            self.handle, self.generatorType, self.seed, self.flags, mu, kappa
        )
        return res

    def zipf(self, alpha, shape=None, dtype=np.uint32):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_zipf(
            self.handle, self.generatorType, self.seed, self.flags, alpha
        )
        return res

    def geometric(self, p, shape=None, dtype=np.uint32):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_geometric(
            self.handle, self.generatorType, self.seed, self.flags, p
        )
        return res

    def wald(self, mean, scale, shape=None, dtype=np.float64):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_wald(
            self.handle, self.generatorType, self.seed, self.flags, mean, scale
        )
        return res

    def binomial(self, ntrials, p, shape=None, dtype=np.uint32):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_binomial(
            self.handle, self.generatorType, self.seed, self.flags, ntrials, p
        )
        return res

    def negative_binomial(self, ntrials, p, shape=None, dtype=np.uint32):
        if shape is None:
            shape = (1,)
        if not isinstance(shape, tuple):
            shape = (shape,)
        res = ndarray(shape, dtype=dtype)
        res._thunk.bitgenerator_negative_binomial(
            self.handle, self.generatorType, self.seed, self.flags, ntrials, p
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
