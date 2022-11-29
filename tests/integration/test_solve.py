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
from utils.comparisons import allclose

import cunumeric as num

SIZES = (8, 9, 255)

RTOL = {
    np.dtype("f"): 1e-1,
    np.dtype("F"): 1e-1,
    np.dtype("d"): 1e-5,
    np.dtype("D"): 1e-5,
}

ATOL = {
    np.dtype("f"): 1e-3,
    np.dtype("F"): 1e-3,
    np.dtype("d"): 1e-8,
    np.dtype("D"): 1e-8,
}


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("a_dtype", ("f", "d", "F", "D"))
@pytest.mark.parametrize("b_dtype", ("f", "d", "F", "D"))
def test_solve_1d(n, a_dtype, b_dtype):
    a = np.random.rand(n, n).astype(a_dtype)
    b = np.random.rand(n).astype(b_dtype)

    out = num.linalg.solve(a, b)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]
    assert allclose(
        b, num.matmul(a, out), rtol=rtol, atol=atol, check_dtype=False
    )


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("a_dtype", ("f", "d", "F", "D"))
@pytest.mark.parametrize("b_dtype", ("f", "d", "F", "D"))
def test_solve_2d(n, a_dtype, b_dtype):
    a = np.random.rand(n, n).astype(a_dtype)
    b = np.random.rand(n, n + 2).astype(b_dtype)

    out = num.linalg.solve(a, b)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]
    assert allclose(
        b, num.matmul(a, out), rtol=rtol, atol=atol, check_dtype=False
    )


def test_solve_corner_cases():
    a = num.random.rand(1, 1)
    b = num.random.rand(1)

    out = num.linalg.solve(a, b)
    assert allclose(b, num.matmul(a, out))

    b = num.random.rand(1, 1)
    out = num.linalg.solve(a, b)
    assert allclose(b, num.matmul(a, out))


def test_solve_b_is_empty():
    a = num.random.rand(1, 1)
    b = num.atleast_2d([])

    out = num.linalg.solve(a, b)
    assert np.array_equal(b, out)


def test_solve_dtype_int():
    n = 8
    a = np.random.randint(0, 10, size=(n, n), dtype="i4")
    b = np.random.randint(0, 10, size=(n,), dtype="i4")

    out = num.linalg.solve(a, b)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]
    assert allclose(
        b, num.matmul(a, out), rtol=rtol, atol=atol, check_dtype=False
    )


def test_solve_with_output():
    n = 8
    a = np.random.rand(n, n).astype("f")
    b = np.random.rand(n).astype("f")
    output = np.zeros((n,)).astype("f")

    out = num.linalg.solve(a, b, out=output)

    rtol = RTOL[out.dtype]
    atol = ATOL[out.dtype]
    assert allclose(
        b, num.matmul(a, out), rtol=rtol, atol=atol, check_dtype=False
    )
    assert allclose(
        b, num.matmul(a, output), rtol=rtol, atol=atol, check_dtype=False
    )


class TestSolveErrors:
    def setup_method(self):
        self.n = 3
        self.a = num.random.rand(self.n, self.n).astype("d")
        self.b = num.random.rand(self.n).astype("d")

    def test_a_bad_dim(self):
        a = num.random.rand(self.n).astype("d")
        msg = "Array must be at least two-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve(a, self.b)

        a = 10
        msg = "Array must be at least two-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve(a, self.b)

    def test_b_bad_dim(self):
        b = 10
        msg = "Array must be at least one-dimensional"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve(self.a, b)

    def test_a_bad_dtype_float16(self):
        a = self.a.astype("e")
        msg = "array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.solve(a, self.b)

    def test_b_bad_dtype_float16(self):
        b = self.b.astype("e")
        msg = "array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.solve(self.a, b)

    def test_a_last_2_dims_not_square(self):
        a = num.random.rand(self.n, self.n + 1).astype("d")
        msg = "Last 2 dimensions of the array must be square"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.solve(a, self.b)

    def test_a_b_mismatched_shape(self):
        b = num.random.rand(self.n + 1).astype("d")
        with pytest.raises(ValueError):
            num.linalg.solve(self.a, b)

        b = num.random.rand(self.n + 1, self.n).astype("d")
        with pytest.raises(ValueError):
            num.linalg.solve(self.a, b)

    def test_output_mismatched_shape(self):
        output = num.zeros((self.n + 1,)).astype("d")
        msg = "Output shape mismatch"
        with pytest.raises(ValueError, match=msg):
            num.linalg.solve(self.a, self.b, out=output)

    def test_output_mismatched_dtype(self):
        output = num.zeros((self.n,)).astype("f")
        msg = "Output type mismatch"
        with pytest.raises(TypeError, match=msg):
            num.linalg.solve(self.a, self.b, out=output)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
