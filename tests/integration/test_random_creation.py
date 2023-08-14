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

import os
from typing import Any, Tuple

import numpy as np
import pytest
from utils.random import assert_distribution

import cunumeric as num

LEGATE_TEST = os.environ.get("LEGATE_TEST", None) == "1"
EAGER_TEST = os.environ.get("CUNUMERIC_FORCE_THUNK", None) == "eager"


def test_randn():
    # We cannot expect that random generation will match with NumPy's,
    # even if initialized with the same seed, so all we can do to verify
    # the results is check that they have the expected distribution.
    a_num = num.random.randn(10000)
    # `mean(a_num)` itself will be normally distributed, with:
    # mean = population mean = 0.0
    # stddev = population stddev / sqrt(samples) = 0.01
    # so a range of -0.05 to 0.05 represents 5 standard deviations
    # which should be extremely unlikely to run over
    assert_distribution(a_num, 0.0, 1.0, mean_tol=0.05)


def reseed_and_gen_random(
    func: str, seed: Any, *args: Any, **kwargs: Any
) -> Tuple[Any, Any]:
    """Reseeed singleton rng and generate random in NumPy and cuNumeric."""
    return gen_random_from_both(func, *args, **kwargs)


def gen_random_from_both(
    func: str, *args: Any, **kwargs: Any
) -> Tuple[Any, Any]:
    """Call the same random function from both NumPy and cuNumeric."""
    return (
        getattr(np.random, func)(*args, **kwargs),
        getattr(num.random, func)(*args, **kwargs),
    )


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(
            12345,
            marks=pytest.mark.xfail(
                not EAGER_TEST,
                reason="cuNumeric does not respect the singleton generator",
            ),
            # https://github.com/nv-legate/cunumeric/issues/601
            # NumPy: generates the same array after initializing with the seed.
            # cuNumeric: keeps generating different arrays.
            # seed is respected in Eager mode.
        ),
        pytest.param(
            None,
            marks=pytest.mark.xfail(
                not num.runtime.has_curand,
                reason="legacy RNG fallback treats seed(None) as seed(0)",
            ),
            # https://github.com/nv-legate/cunumeric/issues/1018
            # NumPy: seed(None) is equivalent to seed(<get_system_random>())
            # cuNumeric non-cuRAND fallback: seed(None) equivalent to seed(0)
        ),
        pytest.param(
            (4, 6, 8),
            marks=pytest.mark.xfail(
                reason="cuNumeric does not take tuple as seed"
            ),
            # NumPy: pass
            # cuNumeric: from runtime.set_next_random_epoch(int(init)):
            # TypeError: int() argument must be a string, a bytes-like object
            # or a real number, not 'tuple'
        ),
    ],
    ids=str,
)
def test_singleton_seed(seed):
    """Test that the singleton seed intitialize the sequence properly."""
    arr_np_1, arr_num_1 = reseed_and_gen_random("random", seed, 10000)
    arr_np_2, arr_num_2 = reseed_and_gen_random("random", seed, 10000)
    assert np.array_equal(arr_np_1, arr_np_2) == np.array_equal(
        arr_num_1, arr_num_2
    )


@pytest.mark.xfail(
    EAGER_TEST,
    reason="cuNumeric does not respect seed in Eager mode",
)
@pytest.mark.parametrize(
    "seed",
    [
        12345,
        pytest.param(
            (0, 4, 5),
            marks=pytest.mark.xfail(
                reason="cuNumeric fails to generate random"
                # NumPy: pass
                # cuNumeric: struct.error: required argument is not an integer
            ),
        ),
    ],
    ids=str,
)
def test_default_rng_seed(seed):
    rng_np_1, rng_num_1 = gen_random_from_both("default_rng", seed)
    rng_np_2, rng_num_2 = gen_random_from_both("default_rng", seed)
    assert rng_np_1.random() == rng_np_2.random()
    assert rng_num_1.random() == rng_num_2.random()


@pytest.mark.xfail(
    EAGER_TEST,
    reason="cuNumeric does not respect seed in Eager mode",
)
@pytest.mark.xfail(
    not num.runtime.has_curand,
    reason="XORWOW not available without cuRAND",
)
def test_default_rng_bitgenerator():
    seed = 12345
    rng_np_1 = np.random.default_rng(np.random.PCG64(seed))
    rng_num_1 = num.random.default_rng(num.random.XORWOW(seed))
    rng_np_2 = np.random.default_rng(np.random.PCG64(seed))
    rng_num_2 = num.random.default_rng(num.random.XORWOW(seed))
    assert rng_np_1.random() == rng_np_2.random()
    assert rng_num_1.random() == rng_num_2.random()


@pytest.mark.xfail(
    EAGER_TEST,
    reason="cuNumeric does not respect seed in Eager mode",
)
def test_default_rng_generator():
    steps = 3
    seed = 12345
    seed_rng_np, seed_rng_num = gen_random_from_both("default_rng", seed)
    np_rngs = [np.random.default_rng(seed_rng_np) for _ in range(steps)]
    num_rngs = [num.random.default_rng(seed_rng_num) for _ in range(steps)]
    np_seq_1 = [rng.random() for rng in np_rngs]
    num_seq_1 = [rng.random() for rng in num_rngs]

    rng_np, rng_num = gen_random_from_both("default_rng", seed)
    np_seq_2 = [rng_np.random() for _ in range(steps)]
    num_seq_2 = [rng_num.random() for _ in range(steps)]
    assert np_seq_1 == np_seq_2
    assert num_seq_1 == num_seq_2


@pytest.mark.parametrize("shape", [(20, 2, 31), (10000,)], ids=str)
def test_rand(shape):
    arr_np, arr_num = gen_random_from_both("rand", *shape)
    assert arr_np.shape == arr_num.shape
    assert_distribution(
        arr_num, np.mean(arr_np), np.std(arr_np), mean_tol=0.05
    )


LOW_HIGH = [
    (5, 10000),
    (-10000, 5),
]
SMALL_RNG_SIZES = [5, 1024, (1, 2)]
LARGE_RNG_SIZES = [10000, (20, 50, 4)]
ALL_RNG_SIZES = SMALL_RNG_SIZES + LARGE_RNG_SIZES + [None]
INT_DTYPES = [np.int64, np.int32, np.int16]
UINT_DTYPES = [np.uint64, np.uint16, np.uint0]
FLOAT_DTYPES = [np.float16, np.float128, np.float64]


@pytest.mark.parametrize("size", ALL_RNG_SIZES, ids=str)
@pytest.mark.parametrize("low, high", LOW_HIGH, ids=str)
@pytest.mark.parametrize("dtype", INT_DTYPES, ids=str)
def test_randint_basic_stats(low, high, size, dtype):
    arr_np, arr_num = gen_random_from_both(
        "randint", low=low, high=high, size=size, dtype=dtype
    )
    assert arr_np.dtype == arr_num.dtype
    assert arr_np.shape == arr_num.shape
    assert np.min(arr_num) >= low
    assert np.max(arr_num) < high


@pytest.mark.parametrize("low", [1024, 1025, 12345], ids=str)
@pytest.mark.parametrize("size", LARGE_RNG_SIZES, ids=str)
@pytest.mark.parametrize("dtype", INT_DTYPES, ids=str)
def test_randint_low_limit_only(low, size, dtype):
    arr_np, arr_num = gen_random_from_both(
        "randint", low, size=size, dtype=dtype
    )
    assert arr_np.dtype == arr_np.dtype
    assert arr_np.shape == arr_num.shape
    assert_distribution(
        arr_num, np.mean(arr_np), np.std(arr_np), mean_tol=0.05
    )


def test_randint_high_limit():
    limit = 10
    arr_np, arr_num = gen_random_from_both("randint", 10, size=100)
    assert np.max(arr_np) < limit
    assert np.max(arr_num) < limit


def test_random_integers_high_limit():
    limit = 10
    arr_np, arr_num = gen_random_from_both("random_integers", 10, size=100)
    assert np.max(arr_np) <= limit
    assert np.max(arr_num) <= limit


@pytest.mark.xfail(reason="cuNumeric raises NotImplementedError")
@pytest.mark.parametrize(
    "low, high", [(3000.45, 15000), (123, 456.7), (12.3, 45.6)], ids=str
)
def test_randint_float_range(low, high):
    # NumPy returns integer scalar
    # cuNumeric raises one of the following
    # NotImplementedError: 'low' must be an integer
    # NotImplementedError: 'high' must be an integer or None
    arr_np, arr_num = gen_random_from_both(
        "randint", low=low, high=high, size=1025
    )
    assert_distribution(
        arr_num, np.mean(arr_np), np.std(arr_np), mean_tol=0.05
    )


@pytest.mark.xfail(
    not num.runtime.has_curand or not EAGER_TEST,
    reason="cuNumeric raises NotImplementedError",
)
@pytest.mark.parametrize("size", ALL_RNG_SIZES, ids=str)
@pytest.mark.parametrize("low, high", [(1000, 65535), (0, 1024)], ids=str)
@pytest.mark.parametrize("dtype", UINT_DTYPES, ids=str)
def test_randint_uint(low, high, dtype, size):
    # NotImplementedError: cunumeric.random.randint must be given an integer
    # dtype
    # NotImplementedError: type for random.integers has to be int64 or int32
    # or int16
    arr_np, arr_num = gen_random_from_both(
        "randint", low=low, high=high, size=size, dtype=dtype
    )
    assert arr_np.dtype == arr_np.dtype
    assert arr_np.shape == arr_num.shape
    assert np.min(arr_num) >= low
    assert np.max(arr_num) <= high


def test_randint_size_none():
    arr_np, arr_num = gen_random_from_both("randint", 1234, size=None)
    assert np.ndim(arr_np) == np.ndim(arr_num)


def test_randint_size_zero():
    arr_np, arr_num = gen_random_from_both("randint", 1234, size=0)
    assert arr_np.dtype == arr_np.dtype
    assert arr_np.shape == arr_np.shape


@pytest.mark.parametrize("low, high", LOW_HIGH, ids=str)
@pytest.mark.parametrize("size", LARGE_RNG_SIZES, ids=str)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_randint_distribution(low, high, size, dtype):
    arr_np, arr_num = gen_random_from_both(
        "randint", low=low, high=high, size=size, dtype=dtype
    )
    assert_distribution(
        arr_num, np.mean(arr_np), np.std(arr_np), mean_tol=0.05
    )


@pytest.mark.xfail(
    not num.runtime.has_curand or not EAGER_TEST,
    reason="cuNumeric raises NotImplementedError",
)
@pytest.mark.parametrize("size", (1024, 1025))
def test_randint_bool(size):
    """Test randint with boolean output dtype."""
    arr_np, arr_num = gen_random_from_both("randint", 2, size=size, dtype=bool)
    assert_distribution(
        arr_num, np.mean(arr_np), np.std(arr_np), mean_tol=0.05
    )
    # NumPy pass
    # cuNumeric not num.runtime.has_curand:
    # NotImplementedError: cunumeric.random.randint must be given an integer
    # dtype
    # cuNumeric LEGATE_TEST=1 or size > 1024:
    # NotImplementedError: type for random.integers has to be int64 or int32
    # or int16


@pytest.mark.parametrize("low, high", LOW_HIGH, ids=str)
@pytest.mark.parametrize("size", LARGE_RNG_SIZES, ids=str)
def test_random_integers(low, high, size):
    arr_np, arr_num = gen_random_from_both(
        "random_integers", low=low, high=high, size=size
    )
    assert arr_np.shape == arr_num.shape
    assert_distribution(
        arr_num, np.mean(arr_np), np.std(arr_np), mean_tol=0.05
    )


@pytest.mark.parametrize("size", SMALL_RNG_SIZES + LARGE_RNG_SIZES, ids=str)
def test_random_sample_basic_stats(size):
    arr_np, arr_num = gen_random_from_both("random_sample", size=size)
    assert arr_np.shape == arr_num.shape


@pytest.mark.xfail(
    reason="NumPy returns scalar, cuNumeric returns 1-dim array"
)
def test_random_sample_size_none():
    arr_np, arr_num = gen_random_from_both("random_sample", size=None)
    assert np.ndim(arr_np) == np.ndim(arr_num)


@pytest.mark.parametrize("size", LARGE_RNG_SIZES, ids=str)
def test_random_sample(size):
    arr_np, arr_num = gen_random_from_both("random_sample", size=size)
    assert_distribution(
        arr_num, np.mean(arr_np), np.std(arr_np), mean_tol=0.05
    )


@pytest.mark.parametrize("size", SMALL_RNG_SIZES, ids=str)
def test_random_std_exponential_basic_stats(size):
    arr_np, arr_num = gen_random_from_both("standard_exponential", size=size)
    assert arr_np.shape == arr_num.shape
    assert arr_np.dtype == arr_num.dtype


@pytest.mark.parametrize("size", SMALL_RNG_SIZES, ids=str)
def test_random_std_gamma_basic_stats(size):
    arr_np, arr_num = gen_random_from_both(
        "standard_gamma", shape=3.1415, size=size
    )
    assert arr_np.shape == arr_num.shape
    assert arr_np.dtype == arr_num.dtype


class TestRandomErrors:
    def assert_exc_from_both(self, func, exc, *args, **kwargs):
        with pytest.raises(exc):
            getattr(np.random, func)(*args, **kwargs)
        with pytest.raises(exc):
            getattr(num.random, func)(*args, **kwargs)

    @pytest.mark.parametrize(
        "seed, expected_exc",
        [
            (-100, ValueError),
            (12.0, TypeError),
            ("abc", TypeError),
        ],
        ids=lambda x: f" {str(getattr(x, 'expected_exception', x))} ",
    )
    @pytest.mark.xfail(reason="NumPy raises exceptions, cuNumeric pass")
    def test_invalid_seed(self, seed, expected_exc):
        self.assert_exc_from_both("seed", expected_exc, seed)
        # -100: NumPy raises ValueError: Seed must be between 0 and 2**32 - 1
        # 12.0: NumPy raises TypeError: Cannot cast scalar from
        # dtype('float64') to dtype('int64') according to the rule 'safe'
        # "abc": TypeError: Cannot cast scalar from dtype('<U3') to
        # dtype('int64') according to the rule 'safe'
        # cuNumeric accepts both -100 and 12.0, raises ValueError on "abc"
        # ValueError: invalid literal for int() with base 10: 'abc'

    @pytest.mark.parametrize(
        "shape, expected_exc",
        [
            (-100, ValueError),
            (12.0, TypeError),
            ("abc", TypeError),
            (None, TypeError),
            ((12345,), TypeError),
        ],
        ids=lambda x: f" {str(getattr(x, 'expected_exception', x))} ",
    )
    def test_rand_invalid_shape(self, shape, expected_exc):
        self.assert_exc_from_both("rand", expected_exc, shape)

    @pytest.mark.parametrize(
        "low, high, expected_exc",
        [
            pytest.param(
                -10000,
                None,
                ValueError,
            ),
            pytest.param(
                -10000,
                -20000,
                ValueError,
            ),
        ],
        ids=str,
    )
    def test_randint_invalid_range(self, low, high, expected_exc):
        self.assert_exc_from_both("randint", expected_exc, low, high)

    @pytest.mark.parametrize(
        "size, expected_exc",
        [
            (12.3, TypeError),
            (-1, ValueError),
            ((12, 4, 5.0), TypeError),
        ],
        ids=str,
    )
    def test_randint_invalid_size(self, size, expected_exc):
        self.assert_exc_from_both("randint", expected_exc, 10000, size=size)

    @pytest.mark.xfail(reason="cuNumeric does not check the bound")
    def test_randint_int16_bound(self):
        # NumPy: ValueError: high is out of bounds for int16
        # cuNumeric: array([13642], dtype=int16)
        expected_exc = ValueError
        self.assert_exc_from_both(
            "randint", expected_exc, 34567, dtype=np.int16
        )

    def test_randint_higher_bound_zero(self):
        expected_exc = ValueError
        self.assert_exc_from_both("randint", expected_exc, 0)

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(
                str,
                marks=pytest.mark.xfail(
                    reason="NumPy raise TypeError, cuNumeric pass"
                ),
            ),
            # NumPy: TypeError: Unsupported dtype dtype('<U') for randint
            # cuNumeric: array(['4'], dtype='<U1')
            pytest.param(
                np.float16,
                marks=pytest.mark.xfail(
                    reason="NumPy: TypeError, cuNumeric: NotImplementedError"
                ),
            ),
            # NumPy: TypeError: Unsupported dtype dtype('float16') for randint
            # cuNumeric with LEGATE_TEST=1: NotImplementedError: type for
            # random.integers has to be int64 or int32 or int16
            # without LEGATE_TEST=1: array([2336.], dtype=float16)
            pytest.param(
                None,
                marks=pytest.mark.xfail(
                    reason="NumPy default to float, cuNumeric pass"
                ),
            ),
            # NumPy: TypeError: Unsupported dtype dtype('float64') for randint
            # cuNumeric: array([401.])
        ],
        ids=str,
    )
    def test_randint_dtype(self, dtype):
        expected_exc = TypeError
        self.assert_exc_from_both("randint", expected_exc, 10000, dtype=dtype)

    @pytest.mark.xfail(reason="cuNumeric pass or raise NotImplementedError")
    @pytest.mark.parametrize("size", (1024, 1025))
    def test_randint_bool(self, size):
        expected_exc = ValueError
        self.assert_exc_from_both(
            "randint", expected_exc, 10000, size=size, dtype=bool
        )
        # NumPy: ValueError: high is out of bounds for bool
        # cuNumeric size > 1024 or LEGATE_TEST=1:
        # NotImplementedError: type for random.integers has to be int64 or
        # int32 or int16
        # cuNumeric size <= 1024 and LEGATE_TEST=0: returns array of booleans

    @pytest.mark.parametrize(
        "size, expected_exc",
        (
            (-1234, ValueError),
            (32.5, TypeError),
            ((9.6, 7), TypeError),
            ("0", TypeError),
        ),
        ids=str,
    )
    def test_random_sample_invalid_size(self, size, expected_exc):
        self.assert_exc_from_both("random_sample", expected_exc, size=size)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
