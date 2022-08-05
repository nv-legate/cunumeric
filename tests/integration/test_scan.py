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

np.random.seed(12345)


def _gen_array(n0, shape, dt, axis, outtype):
    range_lower = 0
    # range 1-5 for ints, avoid zeros for correct testing in prod case
    if np.issubdtype(dt, np.integer):
        range_lower = 1
    if dt == np.complex64:
        A = (
            (5 * np.random.random(shape) + range_lower)
            + (5 * np.random.random(shape) + range_lower) * 1j
        ).astype(np.complex64)
    elif dt == np.complex128:
        A = (
            (5 * np.random.random(shape) + range_lower)
            + (5 * np.random.random(shape) + range_lower) * 1j
        ).astype(np.complex128)
    else:
        A = (5 * np.random.random(shape) + range_lower).astype(dt)
    if n0 == "first_half":
        # second element along all axes is a NAN
        A[(1,) * len(shape)] = np.nan
    elif n0 == "second_half":
        # second from last element along all axes is a NAN
        A[
            tuple(map(lambda i, j: i - j, A.shape, (2,) * len(A.shape)))
        ] = np.nan
    if outtype is None:
        B = None
        C = None
    else:
        if axis is None:
            B = np.zeros(shape=A.size, dtype=outtype)
            C = np.zeros(shape=A.size, dtype=outtype)
        else:
            B = np.zeros(shape=shape, dtype=outtype)
            C = np.zeros(shape=shape, dtype=outtype)
    return A, B, C


def _run_tests(op, n0, shape, dt, axis, out0, outtype):
    print(
        f"Running test: {op}, shape: {shape}, nan location: {n0}"
        f", axis: {axis}, in type: {dt}, out type: {outtype}"
        f", output array not provided: {out0}"
    )
    if out0 is True:
        A, B, C = _gen_array(n0, shape, dt, axis, None)
        B = getattr(num, op)(A, out=None, axis=axis, dtype=outtype)
        C = getattr(np, op)(A, out=None, axis=axis, dtype=outtype)
    else:
        A, B, C = _gen_array(n0, shape, dt, axis, outtype)
        getattr(num, op)(A, out=B, axis=axis, dtype=outtype)
        getattr(np, op)(A, out=C, axis=axis, dtype=outtype)

    print("Checking result...")
    if np.allclose(B, C, equal_nan=True):
        print("PASS!")
    else:
        print("FAIL!")
        print(f"INPUT    : {A}")
        print(f"CUNUMERIC: {B}")
        print(f"NUMPY    : {C}")
        assert False


ops = [
    "cumsum",
    "cumprod",
    "nancumsum",
    "nancumprod",
]
# keeping array sizes small to avoid accumulation variance
# between cunumeric and numpy
shapes = [
    [200],
    [4, 50],
]
axes = [
    None,
    0,
]
out0s = [
    True,
    False,
]
dtypes = [
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]
n0s = [
    None,
    "first_half",
    "second_half",
]


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("axis", axes)
@pytest.mark.parametrize("out0", out0s)
@pytest.mark.parametrize("outtype", dtypes)
@pytest.mark.parametrize("dt", dtypes)
@pytest.mark.parametrize("n0", n0s)
def test_scan(op, shape, axis, out0, outtype, dt, n0):
    if (np.issubdtype(dt, np.integer) and n0 is not None) or (
        np.issubdtype(outtype, np.integer)
        and (op == "cumsum" or op == "cumprod")
        and n0 is not None
    ):
        return
    _run_tests(op, n0, shape, dt, axis, out0, outtype)


def test_empty_inputs():
    in_np = np.ones(10)
    in_np[5:] = 0
    in_num = num.array(in_np).nonzero()[0]
    in_np = in_np.nonzero()[0]

    out_np = np.cumsum(in_np)
    out_num = num.cumsum(in_num)

    assert np.array_equal(out_np, out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
