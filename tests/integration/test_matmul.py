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
        shapeA, shapeB = shapesAB
        A = num.ones(shapeA)
        B = num.ones(shapeB)
        with pytest.raises(ValueError):
            num.matmul(A, B)

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
        shapeA, shapeB = shapesAB
        A = num.ones(shapeA)
        B = num.ones(shapeB)
        with pytest.raises(ValueError):
            num.matmul(A, B)

    def test_invalid_shape_with_scalar(self):
        with pytest.raises(ValueError):
            num.matmul(3, 3)

        with pytest.raises(ValueError):
            num.matmul(3, num.ones((1,)))
        with pytest.raises(ValueError):
            num.matmul(num.ones((1,)), 3)

        with pytest.raises(ValueError):
            num.matmul(3, num.ones((1, 1)))
        with pytest.raises(ValueError):
            num.matmul(num.ones((1,1)), 3)

    @pytest.mark.parametrize(
        "shape", ((2, 3), (3, 4, 3)), ids=lambda shape: f"(shape={shape})"
    )
    def test_out_invalid_shape(self, shape):
        A = num.ones((3, 2, 4))
        B = num.ones((3, 4, 3))
        out = num.zeros(shape)
        with pytest.raises(ValueError):
            num.matmul(A, B, out=out)

    @pytest.mark.xfail
    def test_out_invalid_shape_DIVERGENCE(self):
        # In Numpy, PASS
        # In cuNumeric, raise ValueError
        A = num.ones((3, 2, 4))
        B = num.ones((3, 4, 3))
        shape = (3, 3, 2, 3)
        out = num.zeros(shape)
        num.matmul(A, B, out=out)


    def test_out_invalid_dtype(self):
        A = num.ones((3, 2, 4))
        B = num.ones((3, 4, 3))
        dtype = np.int64
        out = num.zeros((3, 2, 3), dtype=dtype)
        with pytest.raises(TypeError):
            num.matmul(A, B, out=out)

    @pytest.mark.parametrize(
        "casting_dtype",
        (
                ('no', np.float32),
                ('equiv', np.float32),
                ('safe', np.float32),
                ('same_kind', np.int64),
        ),
        ids=lambda casting_dtype: f"(casting_dtype={casting_dtype})",
    )
    def test_invalid_casting_dtype(self, casting_dtype):
        # In Nmupy, it raises numpy.core._exceptions.UFuncTypeError
        # In cuNumeric, it raises TypeError
        casting, dtype = casting_dtype
        A = num.ones((2, 4))
        B = num.ones((4, 3))
        with pytest.raises(TypeError):
            num.matmul(A, B, casting=casting, dtype=dtype)

    @pytest.mark.xfail
    def test_invalid_casting(self):
        # In Numpy, raise ValueError
        # In cuNumeric, pass
        casting = "unknown"
        A = num.ones((2, 4))
        B = num.ones((4, 3))
        with pytest.raises(ValueError):
            num.matmul(A, B, casting=casting)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))