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

import functools

import numpy as np
import pytest
from utils.comparisons import allclose

import cunumeric as num

np.random.seed(143)


def check_result(in_np, out_np, out_num, **isclose_kwargs):
    if in_np.dtype == "e" or out_np.dtype == "e":
        # The mantissa is only 10 bits, 2**-10 ~= 10^(-4)
        # Gives 1e-3 as rtol to provide extra rounding error.
        f16_rtol = 1e-2
        rtol = isclose_kwargs.setdefault("rtol", f16_rtol)
        # make sure we aren't trying to fp16 compare with less precision
        assert rtol >= f16_rtol

    if "negative_test" in isclose_kwargs:
        is_negative_test = isclose_kwargs["negative_test"]
    else:
        is_negative_test = False

    result = (
        allclose(out_np, out_num, **isclose_kwargs)
        and out_np.dtype == out_num.dtype
    )
    if not result and not is_negative_test:
        print("cunumeric failed the test")
        print("Input:")
        print(in_np)
        print(f"dtype: {in_np.dtype}")
        print("NumPy output:")
        print(out_np)
        print(f"dtype: {out_np.dtype}")
        print("cuNumeric output:")
        print(out_num)
        print(f"dtype: {out_num.dtype}")
    return result


def check_op(op_np, op_num, in_np, out_dtype, **check_kwargs):
    in_num = num.array(in_np)

    out_np = op_np(in_np)
    out_num = op_num(in_num)

    assert check_result(in_np, out_np, out_num, **check_kwargs)

    out_np = np.empty(out_np.shape, dtype=out_dtype)
    out_num = num.empty(out_num.shape, dtype=out_dtype)

    op_np(in_np, out=out_np)
    op_num(in_num, out=out_num)

    assert check_result(in_np, out_np, out_num, **check_kwargs)


def get_op_input(
    shape=(4, 5),
    a_min=None,
    a_max=None,
    randint=False,
    offset=None,
    astype=None,
    out_dtype="d",
    replace_zero=None,
    **check_kwargs,
):
    if randint:
        assert a_min is not None
        assert a_max is not None
        in_np = np.random.randint(a_min, a_max, size=shape)
    else:
        in_np = np.random.randn(*shape)
        if offset is not None:
            in_np = in_np + offset
        if a_min is not None:
            in_np = np.maximum(a_min, in_np)
        if a_max is not None:
            in_np = np.minimum(a_max, in_np)
        if astype is not None:
            in_np = in_np.astype(astype)

    if replace_zero is not None:
        in_np[in_np == 0] = replace_zero

    # converts to a scalar if shape is (1,)
    if in_np.ndim == 1 and in_np.shape[0] == 1:
        in_np = in_np[0]

    return in_np


dtypes = (
    "e",
    "f",
    "d",
)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_var_default_shape(dtype, ddof, axis, keepdims):
    np_in = get_op_input(astype=dtype)

    op_np = functools.partial(np.var, ddof=ddof, axis=axis, keepdims=keepdims)
    op_num = functools.partial(
        num.var, ddof=ddof, axis=axis, keepdims=keepdims
    )

    check_op(op_np, op_num, np_in, dtype)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("axis", [None, 0, 1, 2])
@pytest.mark.parametrize("shape", [(10,), (4, 5), (2, 3, 4)])
def test_var_w_shape(dtype, ddof, axis, shape):
    np_in = get_op_input(astype=dtype, shape=shape)

    if axis is not None and axis >= len(shape):
        axis = None

    op_np = functools.partial(np.var, ddof=ddof, axis=axis)
    op_num = functools.partial(num.var, ddof=ddof, axis=axis)

    check_op(op_np, op_num, np_in, dtype)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize(
    "axis",
    [
        None,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (10, 1),
    ],
)
def test_var_corners(dtype, ddof, axis, shape):
    np_in = get_op_input(astype=dtype, shape=shape)

    if axis is not None and axis >= len(shape):
        axis = None

    op_np = functools.partial(np.var, ddof=ddof, axis=axis)
    op_num = functools.partial(num.var, ddof=ddof, axis=axis)

    check_op(op_np, op_num, np_in, dtype)


@pytest.mark.xfail
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize(
    "axis",
    [
        None,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
    ],
)
def test_var_xfail(dtype, ddof, axis, shape):
    np_in = get_op_input(astype=dtype, shape=shape)

    op_np = functools.partial(np.var, ddof=ddof, axis=axis)
    op_num = functools.partial(num.var, ddof=ddof, axis=axis)

    check_op(op_np, op_num, np_in, dtype, negative_test=True)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)

    sys.exit(pytest.main(sys.argv))
