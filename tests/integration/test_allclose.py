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

SCALARS_TRUE_DEFAULT = (
    (0, -1e-8),
    (1e10, 1.00001e10),
    (1 + 1j, 1 + 1.00001j),
)

SCALARS_TRUE_INF = (
    (np.inf, np.inf),
    (-np.inf, -np.inf),
)


@pytest.mark.parametrize(
    "ab",
    SCALARS_TRUE_DEFAULT + SCALARS_TRUE_INF,
    ids=lambda ab: f"(a={ab[0]},b={ab[1]})",
)
def test_scalar_true(ab):
    a, b = ab
    res_np = np.allclose(a, b)
    res_num = num.allclose(a, b)
    assert res_np is True
    assert res_np == res_num


SCALARS_FALSE_DEFAULT = (
    (0, -0.000001),
    (1e10, 1.0001e10),
    (1 + 1j, 1 + 1.0001j),
)

SCALARS_FALSE_INF = ((np.inf, -np.inf),)


@pytest.mark.parametrize(
    "ab",
    SCALARS_FALSE_DEFAULT + SCALARS_FALSE_INF,
    ids=lambda ab: f"(a={ab[0]},b={ab[1]})",
)
def test_scalar_false(ab):
    a, b = ab
    res_np = np.allclose(a, b)
    res_num = num.allclose(a, b)

    assert res_np is False
    assert res_np == res_num


SHPAES = (
    (1,),
    (6,),
    (1, 1),
    (2, 3),
    (2, 3, 4),
)


@pytest.mark.parametrize("shape", SHPAES, ids=lambda shape: f"(shape={shape})")
def test_array_true(shape):
    len_scalars = len(SCALARS_TRUE_DEFAULT)
    size = np.prod(shape)
    array = [SCALARS_TRUE_DEFAULT[i % len_scalars] for i in range(size)]
    a_np = np.array([x[0] for x in array]).reshape(shape)
    b_np = np.array([x[1] for x in array]).reshape(shape)
    a_num = num.array(a_np)
    b_num = num.array(b_np)

    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)

    assert res_np is True
    assert res_np == res_num


@pytest.mark.parametrize("shape", SHPAES, ids=lambda shape: f"(shape={shape})")
def test_array_true_inf(shape):
    tup_scalars_true = SCALARS_TRUE_DEFAULT + SCALARS_TRUE_INF
    len_scalars = len(tup_scalars_true)
    size = np.prod(shape)
    array = [tup_scalars_true[i % len_scalars] for i in range(size)]
    a_np = np.array([x[0] for x in array]).reshape(shape).astype(float)
    b_np = np.array([x[1] for x in array]).reshape(shape).astype(float)
    a_num = num.array(a_np)
    b_num = num.array(b_np)

    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)

    assert res_np is True
    assert res_np == res_num


@pytest.mark.parametrize("shape", SHPAES, ids=lambda shape: f"(shape={shape})")
def test_array_false(shape):
    len_scalars = len(SCALARS_TRUE_DEFAULT)
    size = np.prod(shape)
    array = [SCALARS_TRUE_DEFAULT[i % len_scalars] for i in range(size)]
    array[-1] = SCALARS_FALSE_DEFAULT[0]
    a_np = np.array([x[0] for x in array]).reshape(shape)
    b_np = np.array([x[1] for x in array]).reshape(shape)
    a_num = num.array(a_np)
    b_num = num.array(b_np)

    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)

    assert res_np is False
    assert res_np == res_num


SHPAES_BROADCASTING1 = (
    (1, 3),
    (2, 3),
    (1, 2, 3),
    (2, 2, 3),
)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "shape_b", SHPAES_BROADCASTING1, ids=lambda shape_b: f"(shape_b={shape_b})"
)
def test_boradcast_true1(shape_b):
    # for all cases,
    # In Numpy, it pass
    # In cuNumeric, it raises AttributeError:
    # 'Store' object has no attribute '_broadcast'
    len_scalars = len(SCALARS_TRUE_DEFAULT)

    shape_a = (3,)
    size_a = np.prod(shape_a)
    array_a = [SCALARS_TRUE_DEFAULT[i % len_scalars] for i in range(size_a)]
    a_np = np.array([x[0] for x in array_a]).reshape(shape_a)

    size_b = np.prod(shape_b)
    array_b = [array_a[i % size_a] for i in range(size_b)]
    b_np = np.array([x[1] for x in array_b]).reshape(shape_b)

    a_num = num.array(a_np)
    b_num = num.array(b_np)
    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)

    assert res_np is True
    assert res_np == res_num


SHPAES_BROADCASTING2 = (
    (1,),
    (1, 1),
    (1, 2, 1),
    (2, 2, 1),
)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "shape_b", SHPAES_BROADCASTING2, ids=lambda shape_b: f"(shape_b={shape_b})"
)
def test_boradcast_true2(shape_b):
    # for all cases,
    # In Numpy, it pass
    # In cuNumeric, it raises AttributeError:
    # 'Store' object has no attribute '_broadcast'
    shape_a = (3,)
    size_a = np.prod(shape_a)
    a_np = np.array(
        [SCALARS_TRUE_DEFAULT[0][0] for _ in range(size_a)]
    ).reshape(shape_a)

    size_b = np.prod(shape_b)
    b_np = np.array(
        [SCALARS_TRUE_DEFAULT[0][1] for _ in range(size_b)]
    ).reshape(shape_b)

    a_num = num.array(a_np)
    b_num = num.array(b_np)

    res_np = np.allclose(a_np, b_np)
    res_num = num.allclose(a_num, b_num)

    assert res_np is True
    assert res_np == res_num


@pytest.mark.parametrize(
    "equal_nan", (False, pytest.param(True, marks=pytest.mark.xfail))
)
@pytest.mark.parametrize(
    "arr",
    ([np.nan], [1, 2, np.nan], [[1, 2], [3, np.nan]]),
    ids=lambda arr: f"(arr={arr})",
)
def test_equal_nan_basic(arr, equal_nan):
    # If equal_nan is True,
    # In Numpy, it pass
    # In cuNumeric, it raises NotImplementedError
    res_np = np.allclose(arr, arr, equal_nan=equal_nan)
    res_num = num.allclose(arr, arr, equal_nan=equal_nan)
    assert res_np == res_num


EMPTY_ARRAY_PAIRS = (
    ([], []),
    ([], [[]]),
    ([[]], [[]]),
)


@pytest.mark.parametrize(
    "ab", EMPTY_ARRAY_PAIRS, ids=lambda ab: f"(a={ab[0]},b={ab[1]})"
)
def test_empty_array(ab):
    a, b = ab
    res_np = np.allclose(a, b)
    res_num = num.allclose(a, b)

    assert res_np is True
    assert res_np == res_num


SCALAR_BROADCASTING = (
    (1e10, [1.00001e10]),
    (1e10, [[1.00001e10]]),
)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "ab", SCALAR_BROADCASTING, ids=lambda ab: f"(a={ab[0]},b={ab[1]})"
)
def test_scalar_broadcasting(ab):
    # for all cases,
    # In Numpy, it pass
    # In cuNumeric, it raises AttributeError:
    # 'Store' object has no attribute '_broadcast'
    a, b = ab
    res_np = np.allclose(a, b)
    res_num = num.allclose(a, b)

    assert res_np is True
    assert res_np == res_num


@pytest.mark.parametrize(
    "ab", SCALARS_FALSE_DEFAULT, ids=lambda ab: f"(a={ab[0]},b={ab[1]})"
)
def test_scalar_rtol_atol_true(ab):
    a, b = ab
    rtol = 1e-04
    atol = 1e-06

    res_np = np.allclose(a, b, rtol=rtol, atol=atol)
    res_num = num.allclose(a, b, rtol=rtol, atol=atol)

    assert res_np is True
    assert res_np == res_num


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
