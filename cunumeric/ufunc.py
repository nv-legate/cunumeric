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


def _ufunc(fun, reduce_fn=None):
    class BaseUfunc(object):
        def __new__(cls, a, b, out=None, where=True):
            return fun(a, b, out=out, where=where)

    result = BaseUfunc

    if reduce_fn is not None:

        class ReducibleUfunc(BaseUfunc):
            @staticmethod
            def reduce(a, axis=0, dtype=None, out=None, keepdims=False):
                return reduce_fn(
                    a, axis=axis, dtype=dtype, out=out, keepdims=keepdims
                )

        result = ReducibleUfunc

    return result


add = _ufunc(_add, _sum)

multiply = _ufunc(_mul, _prod)

true_divide = _ufunc(_tdiv)

maximum = _ufunc(_max2, _max)

minimum = _ufunc(_min2, _min)

mod = _ufunc(_mod)

greater = _ufunc(_gt)

greater_equal = _ufunc(_geq)

less = _ufunc(_lt)

less_equal = _ufunc(_leq)

equal = _ufunc(_eq)

not_equal = _ufunc(_neq)
