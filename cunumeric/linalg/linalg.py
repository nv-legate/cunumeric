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
from cunumeric.array import ndarray
from cunumeric.module import sqrt as _sqrt


def cholesky(a):
    """
    Cholesky decomposition.

    Return the Cholesky decomposition, `L * L.H`, of the square matrix `a`,
    where `L` is lower-triangular and .H is the conjugate transpose operator
    (which is the ordinary transpose if `a` is real-valued).  `a` must be
    Hermitian (symmetric if real-valued) and positive-definite. No
    checking is performed to verify whether `a` is Hermitian or not.
    In addition, only the lower-triangular and diagonal elements of `a`
    are used. Only `L` is actually returned.

    Parameters
    ----------
    a : (..., M, M) array_like
        Hermitian (symmetric if all elements are real), positive-definite
        input matrix.

    Returns
    -------
    L : (..., M, M) array_like
        Upper or lower-triangular Cholesky factor of `a`.  Returns a
        matrix object if `a` is a matrix object.

    Notes
    -----
    The current implementation kills the process when the decomposition fails.

    See Also
    --------
    numpy.linalg.cholesky

    Availability
    --------
    GPU, CPU
    """

    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    shape = lg_array.shape
    if len(shape) < 2:
        raise ValueError(
            f"{len(shape)}-dimensional array given. "
            "Array must be at least two-dimensional"
        )
    elif shape[-1] != shape[-2]:
        raise ValueError("Last 2 dimensions of the array must be square")

    if len(shape) > 2:
        raise NotImplementedError(
            "cuNumeric needs to support stacked 2d arrays"
        )
    return lg_array.cholesky()


def norm(x, ord=None, axis=None, keepdims=False):
    lg_array = ndarray.convert_to_cunumeric_ndarray(x)
    if (axis is None and lg_array.ndim == 1) or type(axis) == int:
        # Handle the weird norm cases
        if ord == np.inf:
            return abs(lg_array).max(axis=axis, keepdims=keepdims)
        elif ord == -np.inf:
            return abs(lg_array).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            # Check for where things are not zero and convert to integer
            # for sum
            temp = (lg_array != 0).astype(np.int64)
            return temp.sum(axis=axis, keepdims=keepdims)
        elif ord == 1:
            return abs(lg_array).sum(axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            s = (lg_array.conj() * lg_array).real
            return _sqrt(s.sum(axis=axis, keepdims=keepdims))
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        elif type(ord) == int:
            absx = abs(lg_array)
            absx **= ord
            ret = absx.sum(axis=axis, keepdims=keepdims)
            ret **= 1 / ord
            return ret
        else:
            raise ValueError("Invalid 'ord' argument passed to norm")
    else:
        raise NotImplementedError(
            "cuNumeric needs support for other kinds of norms"
        )
