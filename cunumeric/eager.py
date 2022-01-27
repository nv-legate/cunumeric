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

import numpy as np

from .config import BinaryOpCode, UnaryOpCode, UnaryRedCode
from .thunk import NumPyThunk


class EagerArray(NumPyThunk):
    """This is an eager thunk for describing NumPy computations.
    It is backed by a standard NumPy array that stores the result
    of the computation locally.

    :meta private:
    """

    def __init__(self, runtime, array, parent=None, key=None):
        NumPyThunk.__init__(self, runtime, array.dtype)
        self.array = array
        self.parent = parent
        self.children = None
        self.key = key
        #: if this ever becomes set (to a DeferredArray), we forward all
        #: operations to it
        self.deferred = None
        self.escaped = False

    @property
    def storage(self):
        if self.deferred is None:
            self.to_deferred_array()
        return self.deferred.storage

    @property
    def shape(self):
        return self.array.shape

    def __numpy_array__(self):
        if self.deferred is not None:
            return self.deferred.__numpy_array__()
        # Track when this escapes. If it escapes we have
        # to be more careful in how we do our attach
        self.record_escape()
        return self.array.__array__()

    def record_escape(self):
        if self.parent is None:
            self.escaped = True
        else:
            self.parent.record_escape()

    def check_eager_args(self, *args):
        for arg in args:
            if self.runtime.is_eager_array(arg):
                if arg.deferred is not None:
                    self.to_deferred_array()
                    break
            elif self.runtime.is_deferred_array(arg):
                self.to_deferred_array()
                break
            else:
                raise RuntimeError("bad argument type")

    def to_deferred_array(self):
        """This is a really important method. It will convert a tree of
        eager NumPy arrays into an equivalent tree of deferred arrays that
        are mirrored by an equivalent logical region tree. To be consistent
        we always do this from the root, so once any array in the tree needs
        to be converted then we do it for all of them.
        :meta private:
        """
        # Check to see if we already have our deferred array
        # or whether we need to go up the tree to have it made
        if self.deferred is None:
            if self.parent is None:
                assert self.runtime.is_supported_type(self.array.dtype)
                # We are at the root of the tree so we need to
                # actually make a DeferredArray to use
                if self.array.size == 1:
                    self.deferred = self.runtime.create_scalar(
                        self.array.data,
                        dtype=self.array.dtype,
                        shape=self.shape,
                        wrap=True,
                    )
                else:
                    self.deferred = self.runtime.find_or_create_array_thunk(
                        self.array,
                        share=self.escaped,
                        defer=True,
                    )
            else:
                # Traverse up the tree to make the deferred array
                self.parent.to_deferred_array()
                assert self.deferred is not None
                # No need to traverse down the parent did it for us
                return self.deferred
        else:  # Quick out
            return self.deferred
        # Traverse down for any children that we have
        if self.children is not None:
            assert self.runtime.is_deferred_array(self.deferred)
            for child in self.children:
                child.deferred = self.deferred.get_item(child.key)
            # After we've made all the deferred views for each child then
            # we can traverse down. Do it this way so we can get partition
            # coalescing where possible
            for child in self.children:
                child.to_deferred_array()
        return self.deferred

    def imag(self):
        if self.deferred is not None:
            return self.deferred.imag()
        return EagerArray(self.runtime, self.array.imag)

    def real(self):
        if self.deferred is not None:
            return self.deferred.imag()
        return EagerArray(self.runtime, self.array.real)

    def conj(self):
        if self.deferred is not None:
            return self.deferred.conj()

        return EagerArray(self.runtime, self.array.conj())

    def convolve(self, v, out, mode):
        if self.deferred is not None:
            self.deferred(v, out, mode)
        else:
            if self.ndim == 1:
                out.array = np.convolve(self.array, v.array, mode)
            else:
                from scipy.signal import convolve

                out.array = convolve(self.array, v.array, mode)

    def copy(self, rhs, deep):
        if self.deferred is None:
            self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.copy(rhs, deep=deep)
        else:
            if self.array.size == 1:
                self.array.fill(rhs.array.item())
            elif deep:
                self.array[:] = rhs.array.__deepcopy__(None)
            else:
                self.array[:] = rhs.array

    @property
    def scalar(self):
        if self.deferred is not None:
            return self.deferred.scalar
        return self.array.size == 1

    def get_scalar_array(self):
        if self.deferred is not None:
            return self.deferred.get_scalar_array()
        return self.array.reshape(())

    def _create_indexing_key(self, key):
        if key is None or key is Ellipsis:
            return key
        if isinstance(key, int):
            return key
        if isinstance(key, slice):
            return key
        if isinstance(key, tuple):
            result = ()
            for k in key:
                result += (self._create_indexing_key(k),)
            return result
        assert isinstance(key, NumPyThunk)
        return self.runtime.to_eager_array(key).array

    def get_item(self, key):
        if self.deferred is not None:
            return self.deferred.get_item(key)
        if self._is_advanced_indexing(key):
            index_key = self._create_indexing_key(key)
            out = self.array[index_key]
            result = EagerArray(self.runtime, out)
        else:
            child = self.array[key]
            result = EagerArray(self.runtime, child, parent=self, key=key)
            if self.children is None:
                self.children = list()
            self.children.append(result)
        return result

    def set_item(self, key, value):
        if self.deferred is None:
            self.check_eager_args(value)
        if self.deferred is not None:
            self.deferred.set_item(key, value)
        else:
            if self._is_advanced_indexing(key):
                index_key = self._create_indexing_key(key)
                if isinstance(value, EagerArray):
                    self.array[index_key] = value.array
                else:
                    self.array[index_key] = value
            else:
                if isinstance(value, EagerArray):
                    self.array[key] = value.array
                else:
                    self.array[key] = value

    def reshape(self, newshape, order):
        if self.deferred is not None:
            return self.deferred.reshape(newshape, order)
        child = self.array.reshape(newshape, order=order)
        # See if we are aliased or not
        if child.base is not self.array:
            result = EagerArray(
                self.runtime, child if child.base is None else child.copy()
            )
        else:
            result = EagerArray(
                self.runtime,
                child,
                parent=self,
                key=("reshape", newshape, order),
            )
            if self.children is None:
                self.children = list()
            self.children.append(result)
        return result

    def squeeze(self, axis):
        if self.deferred is not None:
            return self.deferred.squeeze(axis)
        child = self.array.squeeze(axis)
        # Should be aliased with parent region
        assert child.base is not None
        result = EagerArray(
            self.runtime, child, parent=self, key=("squeeze", axis)
        )
        if self.children is None:
            self.children = list()
        self.children.append(result)
        return result

    def swapaxes(self, axis1, axis2):
        if self.deferred is not None:
            return self.deferred.swapaxes(axis1, axis2)
        child = self.array.swapaxes(axis1, axis2)
        # Should be aliased with parent region
        assert child.base is not None
        result = EagerArray(
            self.runtime, child, parent=self, key=("swapaxes", axis1, axis2)
        )
        if self.children is None:
            self.children = list()
        self.children.append(result)
        return result

    def convert(self, rhs, warn=True):
        if self.deferred is None:
            self.check_eager_args(rhs)
        if self.deferred is not None:
            return self.deferred.convert(rhs, warn=warn)
        else:
            if self.array.size == 1:
                self.array.fill(rhs.array.item())
            else:
                self.array[:] = rhs.array

    def fill(self, value):
        if self.deferred is not None:
            self.deferred.fill(value)
        else:
            self.array.fill(value)

    def dot(self, rhs1, rhs2):
        if self.deferred is None:
            self.check_eager_args(rhs1, rhs2)
        if self.deferred is not None:
            self.deferred.dot(rhs1, rhs2)
        else:
            np.dot(rhs1.array, rhs2.array, out=self.array)

    def transpose(self, rhs, axes):
        if self.deferred is None:
            self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.transpose(rhs, axes)
        else:
            if self.array.size == 1:
                self.array.fill(rhs.array.item())
            else:
                self.array[:] = np.transpose(rhs.array, axes)

    def flip(self, rhs, axes):
        if self.deferred is None:
            self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.flip(rhs, axes)
        else:
            self.array = np.flip(rhs.array, axes)

    def contract(
        self,
        lhs_modes,
        rhs1_thunk,
        rhs1_modes,
        rhs2_thunk,
        rhs2_modes,
        mode2extent,
    ):
        if self.deferred is None:
            self.check_eager_args(rhs1_thunk, rhs2_thunk)
        if self.deferred is not None:
            self.deferred.contract(
                lhs_modes,
                rhs1_thunk,
                rhs1_modes,
                rhs2_thunk,
                rhs2_modes,
                mode2extent,
            )
        else:
            np.einsum(
                f"{''.join(rhs1_modes)},{''.join(rhs2_modes)}"
                f"->{''.join(lhs_modes)}",
                rhs1_thunk.array,
                rhs2_thunk.array,
                out=self.array,
            )

    def choose(self, *args, rhs):
        if self.deferred is None:
            self.check_eager_args(*args, rhs)
        if self.deferred is not None:
            self.deferred.choose(
                *args,
                rhs,
            )
        else:
            choices = tuple(c.array for c in args)
            self.array[:] = np.choose(rhs.array, choices, mode="raise")

    def diag(self, rhs, extract, k):
        if self.deferred is None:
            self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.diag(rhs, extract, k)
        else:
            self.array[:] = np.diag(rhs.array, k)

    def eye(self, k):
        if self.deferred is not None:
            self.deferred.eye(k)
        else:
            if self.array.size == 1:
                self.array.fill(1)
            else:
                self.array[:] = np.eye(
                    self.shape[0], self.shape[1], k, dtype=self.dtype
                )

    def arange(self, start, stop, step):
        if self.deferred is not None:
            self.deferred.arange(start, stop, step)
        else:
            self.array = np.arange(start, stop, step, self.dtype)

    def tile(self, rhs, reps):
        if self.deferred is None:
            self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.tile(rhs, reps)
        else:
            self.array[:] = np.tile(rhs.array, reps)

    def bincount(self, rhs, weights=None):
        if self.deferred is None:
            if weights is not None:
                self.check_eager_args(rhs, weights)
            else:
                self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.bincount(rhs, weights=weights)
        else:
            self.array[:] = np.bincount(
                rhs.array,
                weights.array if weights is not None else None,
                minlength=self.array.size,
            )

    def nonzero(self):
        if self.deferred is not None:
            return self.deferred.nonzero()
        else:
            arrays = self.array.nonzero()
            result = ()
            for array in arrays:
                result += (EagerArray(self.runtime, array),)
            return result

    def sort(self, rhs):
        if self.deferred is None:
            self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.sort(rhs)
        else:
            self.array[:] = np.sort(rhs.array)

    def random_uniform(self):
        if self.deferred is not None:
            self.deferred.random_uniform()
        else:
            if self.array.size == 1:
                self.array.fill(np.random.rand())
            else:
                self.array[:] = np.random.rand(*(self.array.shape))

    def random_normal(self):
        if self.deferred is not None:
            self.deferred.random_normal()
        else:
            if self.array.size == 1:
                self.array.fill(np.random.randn())
            else:
                self.array[:] = np.random.randn(*(self.array.shape))

    def random_integer(self, low, high):
        if self.deferred is not None:
            self.deferred.random_integer()
        else:
            if self.array.size == 1:
                self.array.fill(np.random.randint(low, high))
            else:
                self.array[:] = np.random.randint(
                    low, high, size=self.array.shape, dtype=self.array.dtype
                )

    def unary_op(self, op, op_type, rhs, where, args):
        if self.deferred is None:
            if where is not None and isinstance(where, NumPyThunk):
                self.check_eager_args(rhs, where)
            else:
                self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.unary_op(op, op_type, rhs, where, args)
            return
        if op == UnaryOpCode.ABSOLUTE:
            np.absolute(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.ARCCOS:
            np.arccos(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.ARCSIN:
            np.arcsin(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.ARCTAN:
            np.arctan(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.CEIL:
            np.ceil(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.CLIP:
            np.clip(rhs.array, out=self.array, a_min=args[0], a_max=args[1])
        elif op == UnaryOpCode.CONJ:
            np.conj(rhs.array, out=self.array)
        elif op == UnaryOpCode.COPY:
            self.array[:] = rhs.array[:]
        elif op == UnaryOpCode.COS:
            np.cos(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.EXP:
            np.exp(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.EXP2:
            np.exp2(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.FLOOR:
            np.floor(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.IMAG:
            self.array = np.imag(rhs.array)
        elif op == UnaryOpCode.INVERT:
            np.invert(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.ISINF:
            np.isinf(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.ISNAN:
            np.isnan(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.LOG:
            np.log(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.LOG10:
            np.log10(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.LOGICAL_NOT:
            np.logical_not(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.NEGATIVE:
            np.negative(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.REAL:
            self.array = np.real(rhs.array)
        elif op == UnaryOpCode.RINT:
            np.rint(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.SIGN:
            np.sign(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.SIN:
            np.sin(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.SQRT:
            np.sqrt(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.TAN:
            np.tan(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryOpCode.TANH:
            np.tanh(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        else:
            raise RuntimeError("unsupported unary op " + str(op))

    def unary_reduction(self, op, rhs, where, axes, keepdims, args, initial):
        if self.deferred is None:
            self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.unary_reduction(
                op,
                rhs,
                where,
                axes,
                keepdims,
                args,
                initial,
            )
            return
        if op == UnaryRedCode.ALL:
            np.all(
                rhs.array,
                out=self.array,
                axis=axes,
                keepdims=keepdims,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryRedCode.ANY:
            np.any(
                rhs.array,
                out=self.array,
                axis=axes,
                keepdims=keepdims,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryRedCode.ARGMAX:
            assert len(axes) == 1
            np.argmax(rhs.array, out=self.array, axis=axes[0])
        elif op == UnaryRedCode.ARGMIN:
            assert len(axes) == 1
            np.argmin(rhs.array, out=self.array, axis=axes[0])
        elif op == UnaryRedCode.CONTAINS:
            self.array.fill(args[0] in rhs.array)
        elif op == UnaryRedCode.MAX:
            try:
                # Try the new version of this interface for NumPy
                rhs.array.max(
                    axis=axes,
                    out=self.array,
                    keepdims=keepdims,
                    initial=initial,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            except Exception:  # TDB: refine exception
                rhs.array.max(axis=axes, out=self.array, keepdims=keepdims)
        elif op == UnaryRedCode.MIN:
            try:
                # Try the new version of this interface for NumPy
                rhs.array.min(
                    axis=axes,
                    out=self.array,
                    keepdims=keepdims,
                    initial=initial,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            except Exception:  # TDB: refine exception
                rhs.array.min(axis=axes, out=self.array, keepdims=keepdims)
        elif op == UnaryRedCode.PROD:
            try:
                # Try the new version of this interface for NumPy
                np.prod(
                    rhs.array,
                    out=self.array,
                    axis=axes,
                    keepdims=keepdims,
                    initial=initial,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            except Exception:
                np.prod(
                    rhs.array, out=self.array, axis=axes, keepdims=keepdims
                )
        elif op == UnaryRedCode.SUM:
            try:
                # Try the new version of this interface for NumPy
                np.sum(
                    rhs.array,
                    out=self.array,
                    axis=axes,
                    keepdims=keepdims,
                    initial=initial,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            except Exception:
                np.sum(rhs.array, out=self.array, axis=axes, keepdims=keepdims)
        elif op == UnaryRedCode.COUNT_NONZERO:
            self.array[()] = np.count_nonzero(rhs.array, axis=axes)
        else:
            raise RuntimeError("unsupported unary reduction op " + str(op))

    def binary_op(self, op, rhs1, rhs2, where, args):
        if self.deferred is None:
            if where is not None and isinstance(where, NumPyThunk):
                self.check_eager_args(rhs1, rhs2, where)
            else:
                self.check_eager_args(rhs1, rhs2)
        if self.deferred is not None:
            self.deferred.binary_op(op, rhs1, rhs2, where, args)
        else:
            if op == BinaryOpCode.ADD:
                np.add(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.LOGICAL_AND:
                np.logical_and(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.DIVIDE:
                np.divide(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.EQUAL:
                np.equal(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.FLOOR_DIVIDE:
                np.floor_divide(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.GREATER_EQUAL:
                np.greater_equal(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.GREATER:
                np.greater(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            # elif op == BinaryOpCode.SHIFT_LEFT:
            #    np.left_shift(rhs1.array, rhs2.array, out=self.array,
            #            where=where if not isinstance(where, EagerArray)
            #                        else where.array)
            # elif op == BinaryOpCode.SHIFT_RIGHT:
            #    np.right_shift(rhs1.array, rhs2.array, out=self.array,
            #            where=where if not isinstance(where, EagerArray)
            #                        else where.array)
            elif op == BinaryOpCode.MOD:
                np.mod(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.MULTIPLY:
                np.multiply(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.LOGICAL_OR:
                np.logical_or(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.POWER:
                np.power(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.SUBTRACT:
                np.subtract(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.LOGICAL_XOR:
                np.logical_xor(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.LESS_EQUAL:
                np.less_equal(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.LESS:
                np.less(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.MAXIMUM:
                np.maximum(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.MINIMUM:
                np.minimum(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == BinaryOpCode.NOT_EQUAL:
                np.not_equal(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            else:
                raise RuntimeError("unsupported binary op " + str(op))

    def binary_reduction(self, op, rhs1, rhs2, broadcast, args):
        if self.deferred is None:
            self.check_eager_args(rhs1, rhs2)
        if self.deferred is not None:
            self.deferred.binary_reduction(op, rhs1, rhs2, broadcast, args)
        else:
            if op == BinaryOpCode.ALLCLOSE:
                self.array = np.array(
                    np.allclose(
                        rhs1.array, rhs2.array, rtol=args[0], atol=args[1]
                    )
                )
            elif op == BinaryOpCode.EQUAL:
                self.array = np.array(np.array_equal(rhs1.array, rhs2.array))
            else:
                raise RuntimeError(
                    "unsupported binary reduction op " + str(op)
                )

    def where(self, rhs1, rhs2, rhs3):
        if self.deferred is None:
            self.check_eager_args(rhs1, rhs2, rhs3)
        if self.deferred is not None:
            self.deferred.where(rhs1, rhs2, rhs3)
        else:
            self.array[:] = np.where(rhs1.array, rhs2.array, rhs3.array)

    def trilu(self, rhs, k, lower):
        if self.deferred is None:
            self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.trilu(rhs, k, lower)
        else:
            if lower:
                self.array[:] = np.tril(rhs.array, k)
            else:
                self.array[:] = np.triu(rhs.array, k)

    def cholesky(self, src, no_tril):
        if self.deferred is None:
            self.check_eager_args(src)
        if self.deferred is not None:
            self.deferred.cholesky(src)
        else:
            self.array[:] = np.linalg.cholesky(src.array)
