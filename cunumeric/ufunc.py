# Copyright 2021 NVIDIA Corporation
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

# Define ufuns for binary operations
from .module import (
    add as _add,
    amax as _max,
    amin as _min,
    equal as _eq,
    greater as _gt,
    greater_equal as _geq,
    less as _lt,
    less_equal as _leq,
    maximum as _max2,
    minimum as _min2,
    mod as _mod,
    multiply as _mul,
    not_equal as _neq,
    prod as _prod,
    sum as _sum,
    true_divide as _tdiv,
)


class ufunc(object):
    def __init__(self, name, func, red_func=None):
        self._name = name
        self._func = func
        self._red_func = red_func
        self.__doc__ = func.__doc__

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def reduce(
        self,
        array,
        axis=0,
        dtype=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        """
        reduce(array, axis=0, dtype=None, out=None, keepdims=False, initial=<no
        value>, where=True)

        Reduces `array`'s dimension by one, by applying ufunc along one axis.

        For example, add.reduce() is equivalent to sum().

        Parameters
        ----------
        array : array_like
            The array to act on.
        axis : None or int or tuple of ints, optional
            Axis or axes along which a reduction is performed.  The default
            (`axis` = 0) is perform a reduction over the first dimension of the
            input array. `axis` may be negative, in which case it counts from
            the last to the first axis.
        dtype : data-type code, optional
            The type used to represent the intermediate results. Defaults to
            the data-type of the output array if this is provided, or the
            data-type
            of the input array if no output array is provided.
        out : ndarray, None, or tuple of ndarray and None, optional
            A location into which the result is stored. If not provided or
            None, a freshly-allocated array is returned. For consistency with
            ``ufunc.__call__``, if given as a keyword, this may be wrapped in a
            1-element tuple.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the original `array`.
        initial : scalar, optional
            The value with which to start the reduction.  If the ufunc has no
            identity or the dtype is object, this defaults to None - otherwise
            it defaults to ufunc.identity.  If ``None`` is given, the first
            element of the reduction is used, and an error is thrown if the
            reduction is empty.
        where : array_like of bool, optional
            A boolean array which is broadcasted to match the dimensions of
            `array`, and selects elements to include in the reduction. Note
            that for ufuncs like ``minimum`` that do not have an identity
            defined, one has to pass in also ``initial``.

        Returns
        -------
        r : ndarray
            The reduced array. If `out` was supplied, `r` is a reference to it.

        See Also
        --------
        numpy.ufunc.reduce
        """
        if self._red_func is None:
            raise NotImplementedError(
                f"reduction for {self} is not yet implemented"
            )
        return self._red_func(
            array,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def __repr__(self):
        return f"<ufunc {self.name}>"


add = ufunc("add", _add, red_func=_sum)

multiply = ufunc("mul", _mul, red_func=_prod)

true_divide = ufunc("true_divide", _tdiv)

maximum = ufunc("maximum", _max2, red_func=_max)

minimum = ufunc("minimum", _min2, red_func=_min)

mod = ufunc("mod", _mod)

greater = ufunc("greater", _gt)

greater_equal = ufunc("greater_equal", _geq)

less = ufunc("less", _lt)

less_equal = ufunc("less_equal", _leq)

equal = ufunc("equal", _eq)

not_equal = ufunc("not_equal", _neq)
