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

from legate.core import Future, LegateStore, Region

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


class NumPyThunk(LegateStore):
    """This is the base class for NumPy computations. It has methods
    for all the kinds of computations and operations that can be done
    on Legate NumPy ndarrays.

    :meta private:
    """

    def __init__(self, runtime, shape, dtype):
        self.runtime = runtime
        self.shape = shape
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

    def get_scalar_array(self, stacklevel):
        """Get the actual scalar value of the thunk

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

    # Helper methods that are used in several sub-classes
    def _standardize_slice_key(self, key, dim):
        # Wrap around is permitted exactly once
        start = (
            key.start
            if key.start is None or key.start >= 0
            else key.start + self.shape[dim]
        )
        stop = (
            key.stop
            if key.stop is None or key.stop >= 0
            else key.stop + self.shape[dim]
        )
        stride = key.step if key.step is not None else 1
        if start is not None and (start < 0 or start >= self.shape[dim]):
            raise IndexError(
                "index "
                + str(key.start)
                + " is out of bounds for axis "
                + str(dim)
                + " with size "
                + str(self.shape[dim])
            )
        if stop is not None and (stop < 0 or stop > self.shape[dim]):
            raise IndexError(
                "index "
                + str(key.stop)
                + " is out of bounds for axis "
                + str(dim)
                + " with size "
                + str(self.shape[dim])
            )
        if start is None:
            start = 0
        if stop is None:
            stop = self.shape[dim]
        diff = abs(stop - start)
        if stride == 0:
            raise IndexError("step 0 is not allowed for axis " + str(dim))
        if abs(stride) > diff:
            raise IndexError(
                "step "
                + str(stride)
                + " is larger than range for axis "
                + str(dim)
                + " with extent "
                + str(diff)
            )
        # Now turn this into a standard slice
        return slice(start, stop, stride)

    def _standardize_int_key(self, key, dim):
        # Wrap around is permitted exactly once
        if key < -self.shape[dim]:
            raise IndexError(
                "index "
                + str(key)
                + " is out of bounds for axis "
                + str(dim)
                + " with size "
                + str(self.shape[dim])
            )
        if key >= self.shape[dim]:
            raise IndexError(
                "index "
                + str(key)
                + " is out of bounds for axis "
                + str(dim)
                + " with size "
                + str(self.shape[dim])
            )
        # Now turn this into a standard slice
        if key < 0:
            key = self.shape[dim] + key
        return slice(key, key + 1, 1)

    # This method returns a tuple of slices with one slice per dimension
    # and a tuple of integers indicating if the dimension should be
    # collapsed (-1), kept (0), or added (1)
    def _get_view(self, key):
        # See if we have length method
        if not hasattr(key, "__len__"):
            # No length method, see if it is a slice or an integer
            # We're definitely making a view here
            if key is None:
                # Adding a dimension to the array
                view = (slice(0, 1, 1),)
                dim_map = (1,)
                observed_dim = 0
            elif isinstance(key, slice):
                # Normal slice view
                view = (self._standardize_slice_key(key, 0),)
                dim_map = (0,)
                observed_dim = 1
            elif numpy.isscalar(key):
                # Integer means we are removing this dimension
                view = (self._standardize_int_key(key, 0),)
                dim_map = (-1,)
                observed_dim = 1
            elif key is Ellipsis:
                # Build a view and dim map in each dimension
                view = tuple(map(lambda x: slice(0, x, 1), self.shape))
                dim_map = (0,) * self.ndim
                observed_dim = self.ndim
            else:
                raise TypeError(
                    "index must be an int, sequence, Ellipsis, or None"
                )
        else:
            view = ()
            dim_map = ()
            observed_dim = 0
            for dim in xrange(len(key)):
                if key[dim] is None:
                    view += (slice(0, 1, 1),)
                    dim_map += (1,)
                elif isinstance(key[dim], slice):
                    view += (
                        self._standardize_slice_key(key[dim], observed_dim),
                    )
                    dim_map += (0,)
                    observed_dim += 1
                elif numpy.isscalar(key[dim]):
                    view += (
                        self._standardize_int_key(key[dim], observed_dim),
                    )
                    dim_map += (-1,)
                    observed_dim += 1
                elif key[dim] is Ellipsis:
                    # figure out how many dimensions to handle
                    # Everything except None which are new dimensions count
                    ellipsis_keys = (self.ndim - observed_dim) - len(
                        tuple(filter(lambda x: x is not None, key[dim + 1 :]))
                    )
                    if ellipsis_keys > 0:
                        for offset in xrange(ellipsis_keys):
                            view += (slice(0, self.shape[dim + offset], 1),)
                            dim_map += (0,)
                            observed_dim += 1
                else:
                    raise TypeError(
                        "index must be an int, sequence, Ellipsis, or None"
                    )
        # See if we have any leftover dims we need to fill in
        if observed_dim < self.ndim:
            for dim in xrange(observed_dim, self.ndim):
                view += (slice(0, self.shape[dim], 1),)
                dim_map += (0,)
        return view, dim_map

    @staticmethod
    def _get_view_shape(view, dim_map):
        assert len(view) == len(dim_map)
        # Figure out the shape of the new array
        new_shape = ()
        for dim in xrange(len(view)):
            diff = abs(view[dim].stop - view[dim].start)
            assert diff >= abs(view[dim].step)
            extent = (diff + view[dim].step - 1) // abs(view[dim].step)
            # Don't include any collapsed dimensions
            if dim_map[dim] < 0:
                assert extent == 1
                continue
            new_shape += (extent,)
        return new_shape

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
