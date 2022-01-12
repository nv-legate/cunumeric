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

from __future__ import absolute_import, division, print_function

import numpy
import pyarrow

from legate.core import Future, Region


class NumPyThunk(object):
    """This is the base class for NumPy computations. It has methods
    for all the kinds of computations and operations that can be done
    on cuNumeric ndarrays.

    :meta private:
    """

    def __init__(self, runtime, dtype):
        self.runtime = runtime
        self.context = runtime.legate_context
        self.legate_runtime = runtime.legate_runtime
        self.dtype = dtype

    # From Legate Store class
    @property
    def type(self):
        return pyarrow.from_numpy_dtype(self.dtype)

    # From Legate Store class
    @property
    def kind(self):
        if self.ndim == 0:
            return Future
        else:
            return (Region, int)

    @property
    def storage(self):
        """Return the Legion storage primitive for this NumPy thunk"""
        raise NotImplementedError("Implement in derived classes")

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        s = 1
        if self.ndim == 0:
            return s
        for p in self.shape:
            s *= p
        return s

    def __numpy_array__(self, stacklevel):
        """Return a NumPy array that shares storage for this thunk

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def wrap(self, ndarray):
        """Record that this thunk is wrapped by an ndarray

        :meta private:
        """
        pass

    def imag(self, stacklevel):
        """Return a thunk for the imaginary part of this complex array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def real(self, stacklevel):
        """Return a thunk for the real part of this complex array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def copy(self, rhs, deep, stacklevel):
        """Make a copy of the thunk

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    @property
    def scalar(self):
        """If true then this thunk is convertible to a  ()-shape array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def get_scalar_array(self, stacklevel):
        """Get the actual value out of a scalar thunk, as a ()-shape array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def get_item(self, key, stacklevel, view=None, dim_map=None):
        """Get an item from the thunk

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def set_item(self, key, value, stacklevel):
        """Set an item in the thunk

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def reshape(self, newshape, order, stacklevel):
        """Reshape the array using the same backing storage if possible

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def squeeze(self, axis, stacklevel):
        """Remove dimensions of size 1 from the shape of the array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def swapaxes(self, axis1, axis2, stacklevel):
        """Swap two axes in the representation of the array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def convert(self, rhs, stacklevel, warn=True):
        """Convert the data in our thunk to the type in the target array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def fill(self, value, stacklevel):
        """Fill this thunk with the given value

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def dot(self, rhs1, rhs2, stacklevel):
        """Perform a dot operation on our thunk

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def transpose(self, rhs, axes, stacklevel):
        """Perform a transpose operation on our thunk

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def contract(
        self,
        lhs_modes,
        rhs1_thunk,
        rhs1_modes,
        rhs2_thunk,
        rhs2_modes,
        mode2extent,
        stacklevel,
    ):
        """Perform a generalized tensor contraction onto our thunk

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def choose(
        self,
        *args,
        rhs,
        stacklevel=0,
        callsite=None,
    ):
        """Construct an array from an index array and a
            list of arrays to choose from.

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def diag(self, rhs, extract, k, stacklevel):
        """Fill in or extract a diagonal from a matrix

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def eye(self, k, stacklevel):
        """Fill in our thunk with an identity pattern

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def tile(self, rhs, reps, stacklevel):
        """Tile our thunk onto the target

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def bincount(self, rhs, stacklevel, weights=None):
        """Compute the bincount for the array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def nonzero(self, stacklevel):
        """Return a tuple of thunks for the non-zero indices in each "
        "dimension

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def sort(self, rhs, stacklevel):
        """Sort the array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def random_uniform(self, stacklevel):
        """Fill this array with a random uniform distribution

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def random_normal(self, stacklevel):
        """Fill this array with a random normal distribution

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def random_integer(self, low, high, stacklevel):
        """Fill this array with a random integer distribution

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def unary_op(self, op, op_type, rhs, where, args, stacklevel):
        """Perform a unary operation and put the result in the dst
        array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def unary_reduction(
        self, op, redop, rhs, where, axes, keepdims, args, initial, stacklevel
    ):
        """Perform a unary reduction and put the result in the dst
        array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def binary_op(self, op, rhs1, rhs2, where, args, stacklevel):
        """Perform a binary operation with src and put the result in the
        dst array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def binary_reduction(self, op, rhs1, rhs2, broadcast, args, stacklevel):
        """Perform a binary reduction with src and put the result in the
        dst array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def ternary_op(self, op, rhs1, rhs2, rhs3, where, args, stacklevel):
        """Perform a ternary op with one and two and put the result in
        the dst array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def _is_advanced_indexing(self, key, first=True):
        if key is Ellipsis or key is None:  # np.newdim case
            return False
        if numpy.isscalar(key):
            return False
        if isinstance(key, slice):
            return False
        if isinstance(key, tuple):
            for k in key:
                if self._is_advanced_indexing(k, first=False):
                    return True
            return False
        # Any other kind of thing leads to advanced indexing
        return True
