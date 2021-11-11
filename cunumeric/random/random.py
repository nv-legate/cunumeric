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

import numpy as np
import numpy.random as nprandom
from cunumeric.array import ndarray
from cunumeric.runtime import runtime


def seed(init=None):
    if init is None:
        init = 0
    runtime.set_next_random_epoch(int(init))


def rand(*shapeargs):
    if shapeargs is None:
        return nprandom.rand()
    result = ndarray(shapeargs, dtype=np.dtype(np.float64))
    result._thunk.random_uniform(stacklevel=2)
    return result


def randn(*shapeargs):
    if shapeargs is None:
        return nprandom.randn()
    result = ndarray(shapeargs, dtype=np.dtype(np.float64))
    result._thunk.random_normal(stacklevel=2)
    return result


def random(shape=None):
    if shape is None:
        return nprandom.random()
    result = ndarray(shape, dtype=np.dtype(np.float64))
    result._thunk.random_uniform(stacklevel=2)
    return result


def randint(low, high=None, size=None, dtype=None):
    if size is None:
        return nprandom.randint(low=low, high=high, size=size, dtype=dtype)
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = np.dtype(np.int64)
    # TODO: randint must support unsigned integer dtypes as well
    if dtype.kind != "i":
        raise TypeError(
            "cunumeric.random.randint must be given an integer dtype"
        )
    if not isinstance(size, tuple):
        size = (size,)
    result = ndarray(size, dtype=dtype)
    if high is None:
        if low <= 0:
            raise ValueError(
                "bound must be strictly greater than 0 for randint"
            )
        result._thunk.random_integer(low=0, high=low, stacklevel=2)
    else:
        if low >= high:
            raise ValueError(
                "'high' bound must be strictly greater than 'low' "
                "bound for randint"
            )
        result._thunk.random_integer(low=low, high=high, stacklevel=2)
    return result



def uniform(low, high=None, size=None, dtype=None):
    if size is None:
        return nprandom.uniform(low=low, high=high, size=size, dtype=dtype)
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = np.dtype(np.float64)
    # TODO: randint must support unsigned integer dtypes as well
    if dtype.kind != "f":
        raise TypeError(
            "cunumeric.random.uniform must be given an float dtype"
        )
    if not isinstance(size, tuple):
        size = (size,)
    result = ndarray(size, dtype=dtype)
    if high is None:
        
        result._thunk.random_integer(low=0, high=low, stacklevel=2)
    else:
        if low >= high:
            raise ValueError(
                "'high' bound must be strictly greater than 'low' "
            )
        result._thunk.random_uniform(low=low, high=high, stacklevel=2)
    return result
