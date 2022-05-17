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
from cunumeric._ufunc.math import sqrt as _sqrt
from cunumeric.array import add_boilerplate
from cunumeric.module import empty_like, eye, matmul, ndarray


@add_boilerplate("a")
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
    Multiple GPUs, Multiple CPUs
    """
    shape = a.shape
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
    return _cholesky(a)


@add_boilerplate("a")
def matrix_power(a, n):
    """
    Raise a square matrix to the (integer) power `n`.
    For positive integers `n`, the power is computed by repeated matrix
    squarings and matrix multiplications. If ``n == 0``, the identity matrix
    of the same shape as M is returned. If ``n < 0``, the inverse
    is computed and then raised to the ``abs(n)``.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be "powered".
    n : int
        The exponent can be any integer, positive, negative, or zero.

    Returns
    -------
    a**n : (..., M, M) ndarray
        The return value is the same shape and type as `M`;
        if the exponent is positive or zero then the type of the
        elements is the same as those of `M`. If the exponent is
        negative the elements are floating-point.

    See Also
    --------
    numpy.linalg.matrix_power

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Process inputs
    if a.ndim < 2:
        raise ValueError(f"Expected at least 2d array, but got {a.ndim}d")
    if a.shape[-2] != a.shape[-1]:
        raise ValueError("Last 2 dimensions of the array must be square")
    if not isinstance(n, int):
        raise TypeError("exponent must be an integer")

    # Special cases
    if n == 0:
        a = empty_like(a)
        a[...] = eye(a.shape[-2], dtype=a.dtype)
        return a
    elif n == 1:
        return a.copy()

    # Invert if necessary
    if n < 0:
        # TODO: Add this once cunumeric.inv is implemented
        # a = inv(a)
        # n = abs(n)
        raise NotImplementedError("Negative exponent in matrix_power")

    # Fast paths
    if n == 1:
        return a
    elif n == 2:
        return matmul(a, a)
    elif n == 3:
        return matmul(matmul(a, a), a)

    # Use binary decomposition to reduce the number of matrix multiplications.
    # Here, we iterate over the bits of n, from LSB to MSB, raise `a` to
    # increasing powers of 2, and multiply into the result as needed.
    z = result = None
    while n > 0:
        z = a if z is None else matmul(z, z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else matmul(result, z)

    return result


@add_boilerplate("x")
def norm(x, ord=None, axis=None, keepdims=False):
    """
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms, or
    one of an infinite number of vector norms (described below), depending
    on the value of the ord parameter.

    Parameters
    ----------
    x : array_like
        Input array. If axis is None, x must be 1-D or 2-D, unless ord is None.
        If both axis and ord are None, the 2-norm of x.ravel will be returned.
    ord : ``{non-zero int, inf, -inf, ‘fro’, ‘nuc’}``, optional
        Order of the norm (see table under Notes). inf means numpy’s inf
        object. The default is None.
    axis : None or int or tuple[int, int], optional
        If axis is an integer, it specifies the axis of x along which to
        compute the vector norms. If axis is a 2-tuple, it specifies the axes
        that hold 2-D matrices, and the matrix norms of these matrices are
        computed. If axis is None then either a vector norm (when x is 1-D) or
        a matrix norm (when x is 2-D) is returned. The default is None.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one. With this option the result will
        broadcast correctly against the original x.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    Notes
    -----

    See Also
    --------
    numpy.linalg.norm

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if (axis is None and x.ndim == 1) or type(axis) == int:
        # Handle the weird norm cases
        if ord == np.inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -np.inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            # Check for where things are not zero and convert to integer
            # for sum
            temp = (x != 0).astype(np.int64)
            return temp.sum(axis=axis, keepdims=keepdims)
        elif ord == 1:
            return abs(x).sum(axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            s = (x.conj() * x).real
            return _sqrt(s.sum(axis=axis, keepdims=keepdims))
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        elif type(ord) == int:
            absx = abs(x)
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


def _cholesky(a, no_tril=False):
    """Cholesky decomposition.

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
    Multiple GPUs, Multiple CPUs

    """
    input = a
    if input.dtype.kind not in ("f", "c"):
        input = input.astype("float64")
    output = ndarray(
        shape=input.shape,
        dtype=input.dtype,
        inputs=(input,),
    )
    output._thunk.cholesky(input._thunk, no_tril=no_tril)
    return output
