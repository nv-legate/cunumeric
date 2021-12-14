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

from .thunk import NumPyThunk


class LazyArray(NumPyThunk):
    """This is a lazy thunk for describing NumPy computations.
    It is backed by an AST that describes a computation that is
    yet to be performed.

    :meta private:
    """

    def __init__(self, runtime, shape, dtype, thunk):
        NumPyThunk.__init__(self, runtime, shape, dtype)
        self.thunk = thunk

    @property
    def storage(self):
        raise NotImplementedError("Implement in derived classes")

    def __numpy_array__(self, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def imag(self, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def real(self, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def copy(self, rhs, deep, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    @property
    def scalar(self):
        raise NotImplementedError("Implement in derived classes")

    def get_scalar_array(self, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def get_item(self, key, stacklevel, view=None, dim_map=None):
        raise NotImplementedError("Implement in derived classes")

    def set_item(self, key, value, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def reshape(self, newshape, order, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def squeeze(self, axis, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def swapaxes(self, axis1, axis2, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def convert(self, rhs, stacklevel, warn=True):
        raise NotImplementedError("Implement in derived classes")

    def fill(self, value, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def dot(self, rhs1, rhs2, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def transpose(self, rhs, axes, stacklevel):
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
        raise NotImplementedError("Implement in derived classes")

    def choose(
        self,
        *args,
        rhs,
        stacklevel=0,
        callsite=None,
    ):
        raise NotImplementedError("Implement in derived classes")

    def diag(self, rhs, extract, k, stacklevel):
        """Fill in or extract a diagonal from a matrix

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def diagonal(self, rhs, offset, axis1, axis2, extract, stacklevel):
        """Fill in or extract a diagonal from array

        :meta private:
        """
        raise NotImplementedError("Implement in derived classes")

    def eye(self, k, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def tile(self, rhs, reps, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def bincount(self, rhs, stacklevel, weights=None):
        raise NotImplementedError("Implement in derived classes")

    def nonzero(self, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def sort(self, rhs, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def random_uniform(self, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def random_normal(self, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def random_integer(self, low, high, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def unary_op(self, op, op_type, rhs, where, args, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def unary_reduction(
        self, op, redop, rhs, where, axes, keepdims, args, initial, stacklevel
    ):
        raise NotImplementedError("Implement in derived classes")

    def binary_op(self, op, rhs1, rhs2, where, args, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def binary_reduction(self, op, rhs1, rhs2, broadcast, args, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def ternary_op(self, op, rhs1, rhs2, rhs3, where, args, stacklevel):
        raise NotImplementedError("Implement in derived classes")
