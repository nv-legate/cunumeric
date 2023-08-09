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
from legate.core import LEGATE_MAX_DIM
from utils.contractions import (
    check_default,
    check_permutations,
    check_shapes,
    check_types,
)

import cunumeric as num
from cunumeric.utils import matmul_modes


@pytest.mark.parametrize("a_ndim", range(1, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("b_ndim", range(1, LEGATE_MAX_DIM + 1))
def test(a_ndim, b_ndim):
    name = f"matmul({a_ndim} x {b_ndim})"
    modes = matmul_modes(a_ndim, b_ndim)

    def operation(lib, *args, **kwargs):
        return lib.matmul(*args, **kwargs)

    check_default(name, modes, operation)
    check_permutations(name, modes, operation)
    check_shapes(name, modes, operation)
    if a_ndim <= 2 and b_ndim <= 2:
        check_types(name, modes, operation)


class TestMatmulErrors:
    @pytest.mark.parametrize(
        "shapesAB",
        (
            ((2, 4), (2, 3)),
            ((3, 2, 4), (2, 4, 3)),
            ((3, 2, 4), (3, 2, 3)),
        ),
        ids=lambda shapesAB: f"(shapesAB={shapesAB})",
    )
    def test_invalid_shape_dim_greater_than_one(self, shapesAB):
        expected_exc = ValueError
        shapeA, shapeB = shapesAB
        A_np = np.ones(shapeA)
        B_np = np.ones(shapeB)
        A_num = num.ones(shapeA)
        B_num = num.ones(shapeB)
        with pytest.raises(expected_exc):
            np.matmul(A_np, B_np)
        with pytest.raises(expected_exc):
            num.matmul(A_num, B_num)

    @pytest.mark.parametrize(
        "shapesAB",
        (
            ((3, 2), (3,)),
            pytest.param(((4, 1), (3,)), marks=pytest.mark.xfail),
            ((1, 4), (3,)),
            ((3,), (2, 3)),
            ((3,), (4, 1)),
            pytest.param(((3,), (1, 4)), marks=pytest.mark.xfail),
            ((3,), (2,)),
            pytest.param(((3,), (1,)), marks=pytest.mark.xfail),
        ),
        ids=lambda shapesAB: f"(shapesAB={shapesAB})",
    )
    def test_invalid_shape_with_vector(self, shapesAB):
        # For ((4, 1), (3,)), ((3,), (1, 4)), ((3,), (1,)),
        # In Numpy, raise ValueError
        # In cuNumeric, broadcast 1 to 3 and pass
        expected_exc = ValueError
        shapeA, shapeB = shapesAB
        A_np = np.ones(shapeA)
        B_np = np.ones(shapeB)
        A_num = num.ones(shapeA)
        B_num = num.ones(shapeB)
        with pytest.raises(expected_exc):
            np.matmul(A_np, B_np)
        with pytest.raises(expected_exc):
            num.matmul(A_num, B_num)

    def test_invalid_shape_with_scalar(self):
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            np.matmul(3, 3)
        with pytest.raises(expected_exc):
            num.matmul(3, 3)

        with pytest.raises(expected_exc):
            np.matmul(3, np.ones((1,)))
        with pytest.raises(expected_exc):
            num.matmul(3, num.ones((1,)))

        with pytest.raises(expected_exc):
            np.matmul(np.ones((1,)), 3)
        with pytest.raises(expected_exc):
            num.matmul(num.ones((1,)), 3)

        with pytest.raises(expected_exc):
            np.matmul(3, np.ones((1, 1)))
        with pytest.raises(expected_exc):
            num.matmul(3, num.ones((1, 1)))

        with pytest.raises(expected_exc):
            np.matmul(np.ones((1, 1)), 3)
        with pytest.raises(expected_exc):
            num.matmul(num.ones((1, 1)), 3)

    @pytest.mark.parametrize(
        "shape", ((2, 3), (3, 4, 3)), ids=lambda shape: f"(shape={shape})"
    )
    def test_out_invalid_shape(self, shape):
        expected_exc = ValueError
        A_np = np.ones((3, 2, 4))
        B_np = np.ones((3, 4, 3))
        out_np = np.zeros(shape)
        A_num = num.ones((3, 2, 4))
        B_num = num.ones((3, 4, 3))
        out_num = num.zeros(shape)
        with pytest.raises(expected_exc):
            np.matmul(A_np, B_np, out=out_np)
        with pytest.raises(expected_exc):
            num.matmul(A_num, B_num, out=out_num)

    @pytest.mark.xfail
    def test_out_invalid_shape_DIVERGENCE(self):
        # In Numpy, PASS
        # In cuNumeric, raise ValueError
        A = num.ones((3, 2, 4))
        B = num.ones((3, 4, 3))
        shape = (3, 3, 2, 3)
        out = num.zeros(shape)
        num.matmul(A, B, out=out)

    @pytest.mark.parametrize(
        ("dtype", "out_dtype", "casting"),
        ((None, np.int64, "same_kind"), (float, str, "safe")),
        ids=("direct", "intermediate"),
    )
    def test_out_invalid_dtype(self, dtype, out_dtype, casting):
        expected_exc = TypeError
        A_np = np.ones((3, 2, 4))
        B_np = np.ones((3, 4, 3))
        A_num = num.ones((3, 2, 4))
        B_num = num.ones((3, 4, 3))
        out_np = np.zeros((3, 2, 3), dtype=out_dtype)
        out_num = num.zeros((3, 2, 3), dtype=out_dtype)
        with pytest.raises(expected_exc):
            np.matmul(A_np, B_np, dtype=dtype, out=out_np, casting=casting)
        with pytest.raises(expected_exc):
            num.matmul(A_num, B_num, dtype=dtype, out=out_num, casting=casting)

    @pytest.mark.parametrize(
        "casting_dtype",
        (
            ("no", np.float32),
            ("equiv", np.float32),
            ("safe", np.float32),
            ("same_kind", np.int64),
        ),
        ids=lambda casting_dtype: f"(casting_dtype={casting_dtype})",
    )
    def test_invalid_casting_dtype(self, casting_dtype):
        expected_exc = TypeError
        casting, dtype = casting_dtype
        A_np = np.ones((2, 4))
        B_np = np.ones((4, 3))
        A_num = num.ones((2, 4))
        B_num = num.ones((4, 3))
        with pytest.raises(expected_exc):
            np.matmul(A_np, B_np, casting=casting, dtype=dtype)
        with pytest.raises(expected_exc):
            num.matmul(A_num, B_num, casting=casting, dtype=dtype)

    @pytest.mark.parametrize(
        "dtype", (str, pytest.param(float, marks=pytest.mark.xfail)), ids=str
    )
    def test_invalid_casting(self, dtype):
        expected_exc = ValueError
        casting = "unknown"
        A_np = np.ones((2, 4))
        B_np = np.ones((4, 3), dtype=dtype)
        A_num = num.ones((2, 4))
        B_num = num.ones((4, 3), dtype=dtype)
        # In Numpy, raise ValueError
        with pytest.raises(expected_exc):
            np.matmul(A_np, B_np, casting=casting)
        # cuNumeric does not check casting when A and B are of the same dtype
        with pytest.raises(expected_exc):
            num.matmul(A_num, B_num, casting=casting)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
