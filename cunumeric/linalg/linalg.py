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
from cunumeric.array import convert_to_cunumeric_ndarray
from cunumeric.module import dot, empty_like, eye, matmul, ndarray


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

    lg_array = convert_to_cunumeric_ndarray(a)
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
    return _cholesky(lg_array)


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
    a = convert_to_cunumeric_ndarray(a)
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


def multi_dot(arrays, *, out=None):
    """
    Compute the dot product of two or more arrays in a single function call,
    while automatically selecting the fastest evaluation order.
    `multi_dot` chains `dot` and uses optimal parenthesization
    of the matrices.

    Parameters
    ----------
    arrays : sequence of array_like
        If the first argument is 1-D it is treated as a row vector.
        If the last argument is 1-D it is treated as a column vector.
        The other arguments must be 2-D.
    out : ndarray, optional
        Output argument. This must have the same shape and dtype that would be
        returned if it was not used.

    Returns
    -------
    output : ndarray
        Returns the dot product of the supplied arrays.

    See Also
    --------
    numpy.linalg.multi_dot

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    arrays = [convert_to_cunumeric_ndarray(x) for x in arrays]
    if out is not None:
        out = convert_to_cunumeric_ndarray(out, share=True)

    n = len(arrays)
    # optimization only makes sense for len(arrays) > 2
    if n < 2:
        raise ValueError("multi_dot expects at least two arrays")
    elif n == 2:
        return dot(arrays[0], arrays[1], out=out)

    # save original ndim to reshape the result array into the proper form later
    ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
    # Explicitly convert vectors to 2D arrays to keep the logic of the internal
    # _multi_dot_* functions as simple as possible.
    if arrays[0].ndim == 1:
        arrays[0] = arrays[0][np.newaxis, :]
        if out is not None:
            out = out[np.newaxis, ...]
    if arrays[-1].ndim == 1:
        arrays[-1] = arrays[-1][:, np.newaxis]
        if out is not None:
            out = out[..., np.newaxis]
    for x in arrays:
        if x.ndim != 2:
            raise ValueError("Invalid shape for multi_dot input array")
    if out is not None and out.ndim != 2:
        raise ValueError("Invalid shape for multi_dot output array")

    # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2], out=out)
    else:
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot(arrays, order, 0, n - 1, out=out)

    # return proper shape
    if ndim_first == 1 and ndim_last == 1:
        return result.reshape(())  # scalar
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()  # 1-D
    else:
        return result


def _multi_dot_three(A, B, C, out=None):
    """
    Find the best order for three arrays and do the multiplication.
    """
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return dot(dot(A, B), C, out=out)
    else:
        return dot(A, dot(B, C), out=out)


def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    """
    Return a `np.array` that encodes the optimal order of mutiplications.
    The optimal order array is then used by `_multi_dot()` to do the
    multiplication.
    Also return the cost matrix if `return_costs` is `True`
    The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
    Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.
        cost[i, j] = min([
            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
            for k in range(i, j)])
    """
    n = len(arrays)
    # p stores the dimensions of the matrices
    # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    # m is a matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = np.zeros((n, n), dtype=np.float64)
    # s is the actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = np.empty((n, n), dtype=np.int64)

    for l_ in range(1, n):
        for i in range(n - l_):
            j = i + l_
            m[i, j] = np.Inf
            for k in range(i, j):
                q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index

    return (s, m) if return_costs else s


def _multi_dot(arrays, order, i, j, out=None):
    """Actually do the multiplication with the given order."""
    if i == j:
        # the initial call with non-None out should never get here
        assert out is None

        return arrays[i]
    else:
        return dot(
            _multi_dot(arrays, order, i, order[i, j]),
            _multi_dot(arrays, order, order[i, j] + 1, j),
            out=out,
        )


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

    lg_array = convert_to_cunumeric_ndarray(x)
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
