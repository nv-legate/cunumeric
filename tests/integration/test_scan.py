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
    # range 1-10, avoiding zeros to ensure correct testing for int prod case
    if dt == np.complex64:
        A = (
            (99 * np.random.random(shape) + 1)
            + (99 * np.random.random(shape) + 1) * 1j
        ).astype(np.complex64)
    elif dt == np.complex128:
        A = (
            (99 * np.random.random(shape) + 1)
            + (99 * np.random.random(shape) + 1) * 1j
        ).astype(np.complex128)
    else:
        A = (99 * np.random.random(shape) + 1).astype(dt)
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
    [400],
    [4, 100],
    [4, 10, 10],
]
axes = [
    None,
    0,
]
out0s = [
    True,
    False,
]


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("axis", axes)
@pytest.mark.parametrize("out0", out0s)
def test_scan(op, shape, axis, out0):
    n0s = [
        None,
        "first_half",
        "second_half",
    ]
    int_types = [
        np.int16,
        np.int32,
        np.int64,
    ]
    float_types = [
        np.float32,
        np.float64,
    ]
    complex_types = [
        np.complex64,
        np.complex128,
    ]
    for outtype in int_types:
        for dt in int_types:
            _run_tests(op, None, shape, dt, axis, out0, outtype)
        for dt in float_types:
            for n0 in n0s:
                print("Float to int NAN conversion currently not supported!")
        for dt in complex_types:
            for n0 in n0s:
                print("Complex to int NAN conversion currently not supported!")

    for outtype in float_types:
        for dt in int_types:
            _run_tests(op, None, shape, dt, axis, out0, outtype)
        for dt in float_types:
            for n0 in n0s:
                _run_tests(op, n0, shape, dt, axis, out0, outtype)
        for dt in complex_types:
            for n0 in n0s:
                _run_tests(op, n0, shape, dt, axis, out0, outtype)

    for outtype in complex_types:
        for dt in int_types:
            _run_tests(op, None, shape, dt, axis, out0, outtype)
        for dt in float_types:
            for n0 in n0s:
                _run_tests(op, n0, shape, dt, axis, out0, outtype)
        for dt in complex_types:
            for n0 in n0s:
                _run_tests(op, n0, shape, dt, axis, out0, outtype)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
