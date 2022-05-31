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
from __future__ import annotations

import numpy as np
from cunumeric._ufunc.math import add, sqrt as _sqrt
from cunumeric.array import add_boilerplate, convert_to_cunumeric_ndarray
from cunumeric.module import dot, empty_like, eye, matmul, ndarray
from numpy.core.multiarray import normalize_axis_index


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


# This implementation is adapted closely from NumPy
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


# This implementation is adapted closely from NumPy
def multi_dot(arrays, *, out=None):
    """
    Compute the dot product of two or more arrays in a single function call,
    while automatically selecting the fastest evaluation order.
    `multi_dot` chains `dot` and uses optimal parenthesization
    of the matrices.

    Parameters
    ----------
    arrays : Sequence[array_like]
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


# This implementation is adapted closely from NumPy
@add_boilerplate("x")
def norm(x, ord=None, axis=None, keepdims=False):
    """
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : array_like
        Input array.  If `axis` is None, `x` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``x.ravel`` will be returned.
    ord : ``{non-zero int, inf, -inf, 'fro', 'nuc'}``, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object. The default is None.
    axis : None or int or tuple[int, int], optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default
        is None.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    Notes
    -----
    For values of ``ord < 1``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Both the Frobenius and nuclear norm orders are only defined for
    matrices and raise a ValueError when ``x.ndim != 2``.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    See Also
    --------
    numpy.linalg.norm

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    # Immediately handle some default, simple, fast, and common cases.
    if axis is None:
        ndim = x.ndim
        if (
            (ord is None)
            or (ord in ("f", "fro") and ndim == 2)
            or (ord == 2 and ndim == 1)
        ):

            x = x.ravel()
            if x.dtype.kind == "c":
                x_real = x.real
                x_imag = x.imag
                sqnorm = dot(x_real, x_real) + dot(x_imag, x_imag)
            else:
                sqnorm = dot(x, x)
            ret = _sqrt(sqnorm)
            if keepdims:
                ret = ret.reshape(ndim * [1])
            return ret

    # Normalize the `axis` argument to a tuple.
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    else:
        if not isinstance(axis, tuple):
            axis = (axis,)
        for ax in axis:
            if not isinstance(ax, int):
                raise TypeError(
                    "`axis` must be None, an integer or a tuple of integers"
                )

    if len(axis) == 1:
        if ord == np.inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -np.inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            # Zero norm
            return (x != 0).astype(np.int64).sum(axis=axis, keepdims=keepdims)
        elif ord == 1:
            # special case for speedup
            return add.reduce(abs(x), axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            # special case for speedup
            s = (x.conj() * x).real
            return _sqrt(add.reduce(s, axis=axis, keepdims=keepdims))
        # None of the str-type keywords for ord ("fro", "nuc")
        # are valid for vectors
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            absx = abs(x)
            absx **= ord
            ret = add.reduce(absx, axis=axis, keepdims=keepdims)
            ret **= 1 / ord
            return ret
    elif len(axis) == 2:
        row_axis, col_axis = axis
        row_axis = normalize_axis_index(row_axis, nd)
        col_axis = normalize_axis_index(col_axis, nd)
        if row_axis == col_axis:
            raise ValueError("Duplicate axes given")
        if ord == 2:
            raise NotImplementedError("2-norm requires SVD decomposition")
        elif ord == -2:
            raise NotImplementedError("-2-norm requires SVD decomposition")
        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).max(axis=col_axis)
        elif ord == np.inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).min(axis=col_axis)
        elif ord == -np.inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).min(axis=row_axis)
        elif ord in [None, "fro", "f"]:
            squares = (x.conj() * x).real
            ret = _sqrt(squares.sum(axis=col_axis).sum(axis=row_axis))
        elif ord == "nuc":
            raise NotImplementedError(
                "nuclear norm requires SVD decomposition"
            )
        else:
            raise ValueError("Invalid norm order for matrices")
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        raise ValueError("Improper number of dimensions to norm")


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
