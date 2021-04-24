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
    multiply as _mul,
    not_equal as _neq,
    prod as _prod,
    sum as _sum,
    true_divide as _tdiv,
)


# Base ufunc class
class ufunc(object):
    @staticmethod
    def reduce_impl(
        a, reduceOpString, axis=0, dtype=None, out=None, keepdims=False
    ):
        raise NotImplementedError("reduce ufunc")


# ufunc-add class
class add(ufunc):
    def __new__(cls, a, b, out=None, where=True):
        return _add(a, b, out=out, where=where, stacklevel=2)

    @staticmethod
    def reduce(a, axis=0, dtype=None, out=None, keepdims=False):
        return _sum(
            a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, stacklevel=2
        )


# ufunc-multiply class
class multiply(ufunc):
    def __new__(cls, a, b, out=None, where=True):
        return _mul(a, b, out=out, where=where, stacklevel=2)

    @staticmethod
    def reduce(a, axis=0, dtype=None, out=None, keepdims=False):
        return _prod(
            a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, stacklevel=2
        )


# ufunc-true_divide class
class true_divide(ufunc):
    def __new__(cls, a, b, out=None, where=True):
        return _tdiv(a, b, out=out, where=where, stacklevel=2)


# ufunc-maximum class
class maximum(ufunc):
    def __new__(cls, a, b, out=None, where=True):
        return _max2(a, b, out=out, where=where, stacklevel=2)

    @staticmethod
    def reduce(a, axis=0, dtype=None, out=None, keepdims=False):
        assert dtype is None
        return _max(a, axis=axis, out=out, keepdims=keepdims, stacklevel=2)


# ufunc-minimum class
class minimum(ufunc):
    def __new__(cls, a, b, out=None, where=True):
        return _min2(a, b, out=out, where=where, stacklevel=2)

    @staticmethod
    def reduce(a, axis=0, dtype=None, out=None, keepdims=False):
        assert dtype is None
        return _min(a, axis=axis, out=out, keepdims=keepdims, stacklevel=2)


# ufunc-greater class
class greater(ufunc):
    def __new__(cls, a, b, out=None, where=True):
        return _gt(a, b, out=out, where=where, stacklevel=2)


# ufunc-greater_equal class
class greater_equal(ufunc):
    def __new__(cls, a, b, out=None, where=True):
        return _geq(a, b, out=out, where=where, stacklevel=2)


# ufunc-less class
class less(ufunc):
    def __new__(cls, a, b, out=None, where=True):
        return _lt(a, b, out=out, where=where, stacklevel=2)


# ufunc-less_equal class
class less_equal(ufunc):
    def __new__(cls, a, b, out=None, where=True):
        return _leq(a, b, out=out, where=where, stacklevel=2)


# ufunc-equal class
class equal(ufunc):
    def __new__(cls, a, b, out=None, where=True):
        return _eq(a, b, out=out, where=where, stacklevel=2)


# ufunc-not_equal class
class not_equal(ufunc):
    def __new__(cls, a, b, out=None, where=True):
        return _neq(a, b, out=out, where=where, stacklevel=2)
