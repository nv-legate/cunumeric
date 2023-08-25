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

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import numpy.random as nprandom

from cunumeric.array import ndarray
from cunumeric.runtime import runtime

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..types import NdShapeLike


def seed(init: Union[int, None] = None) -> None:
    if init is None:
        init = 0
    runtime.set_next_random_epoch(int(init))


def rand(*shapeargs: int) -> Union[float, ndarray]:
    """
    rand(d0, d1, ..., dn)

    Random values in a given shape.

    Create an array of the given shape and populate it with random samples from
    a uniform distribution over ``[0, 1)``.

    Parameters
    ----------
    d0, d1, ..., dn : int, optional
        The dimensions of the returned array, must be non-negative.
        If no argument is given a single Python float is returned.

    Returns
    -------
    out : ndarray, shape ``(d0, d1, ..., dn)``
        Random values.

    See Also
    --------
    numpy.random.rand

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if shapeargs == ():
        return nprandom.rand()
    result = ndarray(shapeargs, dtype=np.dtype(np.float64))
    result._thunk.random_uniform()
    return result


def randint(
    low: int,
    high: Union[int, None] = None,
    size: Union[NdShapeLike, None] = None,
    dtype: Union[np.dtype[Any], type, None] = int,
) -> Union[int, ndarray, npt.NDArray[Any]]:
    """
    Return random integers from `low` (inclusive) to `high` (exclusive).

    Parameters
    ----------
    low : int or array_like[int]
        Lowest (signed) integers to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is one above the
        *highest* such integer).
    high : int or array_like[int], optional
        If provided, one above the largest (signed) integer to be drawn
        from the distribution (see above for behavior if ``high=None``).
        If array-like, must contain integer values
    size : int or tuple[int], optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    dtype : data-type, optional
        Desired dtype of the result. Byteorder must be native.
        The default value is int.

    Returns
    -------
    out : int or ndarray[int]
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    See Also
    --------
    numpy.random.randint

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if not isinstance(low, int):
        raise NotImplementedError("'low' must be an integer")
    if high is not None and not isinstance(high, int):
        raise NotImplementedError("'high' must be an integer or None")

    if size is None:
        return nprandom.randint(low=low, high=high, size=size, dtype=dtype)

    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = np.dtype(np.int64)
    # TODO: randint must support unsigned integer dtypes as well
    if dtype.kind != "i":
        raise NotImplementedError(
            "cunumeric.random.randint must be given an integer dtype"
        )
    if isinstance(size, int):
        size = (size,)
    result = ndarray(size, dtype=dtype)
    if high is None:
        if low <= 0:
            raise ValueError(
                "bound must be strictly greater than 0 for randint"
            )
        result._thunk.random_integer(low=0, high=low)
    else:
        if low >= high:
            raise ValueError(
                "'high' bound must be strictly greater than 'low' "
                "bound for randint"
            )
        result._thunk.random_integer(low=low, high=high)
    return result


def randn(*shapeargs: int) -> Union[float, ndarray]:
    """
    randn(d0, d1, ..., dn)

    Return a sample (or samples) from the "standard normal" distribution.

    Parameters
    ----------
    d0, d1, ..., dn : int, optional
        The dimensions of the returned array, must be non-negative.
        If no argument is given a single Python float is returned.

    Returns
    -------
    Z : ndarray or float
        A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from
        the standard normal distribution, or a single such float if
        no parameters were supplied.

    See Also
    --------
    numpy.random.randn

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if shapeargs == ():
        return nprandom.randn()
    result = ndarray(shapeargs, dtype=np.dtype(np.float64))
    result._thunk.random_normal()
    return result


def random(size: Union[NdShapeLike, None] = None) -> Union[float, ndarray]:
    """
    random(size=None)

    Return random floats in the half-open interval [0.0, 1.0).

    See Also
    --------
    numpy.random.random

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if size is None:
        return nprandom.random()
    result = ndarray(size, dtype=np.dtype(np.float64))
    result._thunk.random_uniform()
    return result
