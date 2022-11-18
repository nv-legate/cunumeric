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


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
