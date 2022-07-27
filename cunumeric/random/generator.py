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

    def random(self, size=None, dtype=np.float64, out=None):
        if out is not None:
            if size is not None:
                assert out.shape == size
            assert out.dtype == dtype
        return self.bit_generator.random(size, dtype, out)

    def lognormal(self, mean=0.0, sigma=1.0, size=None, dtype=np.float64):
        return self.bit_generator.lognormal(mean, sigma, size, dtype)

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=np.float64):
        return self.bit_generator.normal(
            mean=loc, sigma=scale, shape=size, dtype=dtype
        )

    def uniform(self, low=0.0, high=1.0, size=None, dtype=np.float64):
        return self.bit_generator.uniform(low, high, size, dtype)

    def poisson(self, lam=1.0, size=None):
        return self.bit_generator.poisson(lam, size)

    def exponential(self, scale=1.0, size=None, dtype=np.float64):
        return self.bit_generator.exponential(
            scale=scale, shape=size, dtype=dtype
        )

    def gumbel(self, loc=0.0, scale=1.0, size=None, dtype=np.float64):
        return self.bit_generator.gumbel(
            mu=loc, beta=scale, shape=size, dtype=dtype
        )

    def laplace(self, loc=0.0, scale=1.0, size=None, dtype=np.float64):
        return self.bit_generator.laplace(
            mu=loc, beta=scale, shape=size, dtype=dtype
        )

    def logistic(self, loc=0.0, scale=1.0, size=None, dtype=np.float64):
        return self.bit_generator.logistic(
            mu=loc, beta=scale, shape=size, dtype=dtype
        )

    def pareto(self, a, size=None, dtype=np.float64):
        return self.bit_generator.pareto(alpha=a, shape=size, dtype=dtype)

    def power(self, a, size=None, dtype=np.float64):
        return self.bit_generator.power(alpha=a, shape=size, dtype=dtype)

    def rayleigh(self, scale, size=None, dtype=np.float64):
        return self.bit_generator.rayleigh(
            sigma=scale, shape=size, dtype=dtype
        )

    def cauchy(self, x0, gamma, size=None, dtype=np.float64):
        return self.bit_generator.cauchy(
            x0=x0, gamma=gamma, shape=size, dtype=dtype
        )

    def standard_cauchy(self, size=None, dtype=np.float64):
        return self.cauchy(0.0, 1.0, size, dtype)

    def standard_exponential(self, size=None, dtype=np.float64):
        return self.exponential(1.0, size, dtype)

    def triangular(self, left, mode, right, size=None, dtype=np.float64):
        return self.bit_generator.triangular(
            a=left, b=right, c=mode, shape=size, dtype=dtype
        )

    def weibull(self, a, size=None, dtype=np.float64):
        return self.bit_generator.weibull(lam=1, k=a, shape=size, dtype=dtype)

    def bytes(self, length):
        return self.bit_generator.bytes(length=length)

    def beta(self, a, b, size=None, dtype=np.float64):
        return self.bit_generator.beta(a=a, b=b, shape=size, dtype=dtype)

    def f(self, dfnum, dfden, size=None, dtype=np.float64):
        return self.bit_generator.f(
            dfnum=dfnum, dfden=dfden, shape=size, dtype=dtype
        )

    def logseries(self, p, size=None, dtype=np.uint32):
        return self.bit_generator.logseries(p=p, shape=size, dtype=np.uint32)

    def noncentral_f(self, dfnum, dfden, nonc, size=None, dtype=np.float64):
        return self.bit_generator.noncentral_f(
            dfnum=dfnum, dfden=dfden, nonc=nonc, shape=size, dtype=dtype
        )

    def chisquare(self, df, size=None, dtype=np.float64):
        return self.bit_generator.chisquare(
            df=df, nonc=0.0, shape=size, dtype=dtype
        )

    def noncentral_chisquare(self, df, nonc, size=None, dtype=np.float64):
        return self.bit_generator.chisquare(
            df=df, nonc=nonc, shape=size, dtype=dtype
        )

    def gamma(self, shape, scale=1.0, size=None, dtype=np.float64):
        return self.bit_generator.gamma(
            k=shape, theta=scale, shape=size, dtype=dtype
        )

    def standard_gamma(self, shape, size=None, dtype=np.float64):
        return self.gamma(shape=shape, scale=1.0, size=size, dtype=dtype)

    def standard_t(self, df, size=None, dtype=np.float64):
        return self.bit_generator.standard_t(df=df, shape=size, dtype=dtype)

    def hypergeometric(self, ngood, nbad, nsample, size=None, dtype=np.uint32):
        return self.bit_generator.hypergeometric(
            ngood=ngood, nbad=nbad, nsample=nsample, shape=size, dtype=dtype
        )

    def vonmises(self, mu, kappa, size=None, dtype=np.float64):
        return self.bit_generator.vonmises(
            mu=mu, kappa=kappa, shape=size, dtype=dtype
        )

    def zipf(self, a, size=None, dtype=np.uint32):
        return self.bit_generator.zipf(alpha=a, shape=size, dtype=dtype)

    def geometric(self, p, size=None, dtype=np.uint32):
        return self.bit_generator.geometric(p=p, shape=size, dtype=dtype)


def default_rng(seed=None):
    if seed is None:
        return Generator(XORWOW())
    elif isinstance(seed, BitGenerator):
        return Generator(seed)
    elif isinstance(seed, Generator):
        return seed
    else:
        return Generator(XORWOW(seed))
