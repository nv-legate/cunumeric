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

from .config import NumPyOpCode
from .thunk import NumPyThunk


class EagerArray(NumPyThunk):
    """This is an eager thunk for describing NumPy computations.
    It is backed by a standard NumPy array that stores the result
    of the computation locally.

    :meta private:
    """

    def __init__(self, runtime, array, parent=None, key=None, shadow=False):
        NumPyThunk.__init__(self, runtime, array.shape, array.dtype)
        self.array = array
        self.parent = parent
        self.children = None
        self.key = key
        self.deferred = None
        self.shadow = shadow
        self.escaped = False

    @property
    def storage(self):
        if self.deferred is None:
            self.to_deferred_array(stacklevel=2)
        return self.deferred.storage

    def __numpy_array__(self, stacklevel):
        if self.deferred is not None:
            return self.deferred.__numpy_array__(stacklevel=(stacklevel + 1))
        # Track when this escapes. If it escapes we have
        # to be more careful in how we do our attach
        self.record_escape()
        return self.array

    def record_escape(self):
        if self.parent is None:
            self.escaped = True
        else:
            self.parent.record_escape()

    def check_eager_args(self, stacklevel, *args):
        for arg in args:
            if self.runtime.is_eager_array(arg):
                if arg.deferred is not None:
                    self.to_deferred_array(stacklevel=(stacklevel + 1))
                    break
            elif self.runtime.is_deferred_array(arg):
                self.to_deferred_array(stacklevel=(stacklevel + 1))
                break
            else:
                raise RuntimeError("bad argument type")

    def to_deferred_array(self, stacklevel):
        """This is a really important method. It will convert a tree of
        eager NumPy arrays into an equivalent tree of deferred arrays that
        are mirrored by an equivalent logical region tree. To be consistent
        we always do this from the root, so once any array in the tree needs
        to be converted then we do it for all of them.
        :meta private:
        """
        assert not self.shadow
        # Check to see if we already have our deferred array
        # or whether we need to go up the tree to have it made
        if self.deferred is None:
            if self.parent is None:
                assert self.runtime.is_supported_type(self.array.dtype)
                # We are at the root of the tree so we need to
                # actually make DeferredArray to use
                if self.array.size == 1:
                    self.deferred = self.runtime.create_future(
                        self.array.data,
                        self.array.nbytes,
                        wrap=True,
                        dtype=self.array.dtype,
                    )
                else:
                    self.deferred = self.runtime.find_or_create_array_thunk(
                        self.array,
                        stacklevel=(stacklevel + 1),
                        share=self.escaped,
                        defer=True,
                    )
            else:
                # Traverse up the tree to make the deferred array
                self.parent.to_deferred_array(stacklevel=(stacklevel + 1))
                assert self.deferred is not None
                # No need to traverse down the parent did it for us
                return self.deferred
        else:  # Quick out
            return self.deferred
        # Traverse down for any children that we have
        if self.children is not None:
            assert self.runtime.is_deferred_array(self.deferred)
            for child in self.children:
                child.deferred = self.deferred.get_item(
                    child.key, stacklevel=(stacklevel + 1)
                )
            # After we've made all the deferred views for each child then
            # we can traverse down. Do it this way so we can get partition
            # coalescing where possible
            for child in self.children:
                child.to_deferred_array(stacklevel=(stacklevel + 1))
        return self.deferred

    def imag(self, stacklevel):
        if self.deferred is not None:
            return self.deferred.imag(stacklevel=(stacklevel + 1))
        return self.array.imag()

    def real(self, stacklevel):
        if self.deferred is not None:
            return self.deferred.imag(stacklevel=(stacklevel + 1))
        return self.array.real()

    def copy(self, rhs, deep, stacklevel):
        if self.shadow:
            rhs = self.runtime.to_eager_array(rhs, stacklevel=(stacklevel + 1))
        elif self.deferred is None:
            self.check_eager_args((stacklevel + 1), rhs)
        if self.deferred is not None:
            self.deferred.copy(rhs, deep=deep, stacklevel=(stacklevel + 1))
        else:
            if self.array.size == 1:
                if deep:
                    self.array.fill(rhs.array.item().__deepcopy__(None))
                else:
                    self.array.fill(rhs.array.item())
            else:
                if deep:
                    self.array[:] = rhs.array.__deepcopy__(None)
                else:
                    self.array[:] = rhs.array
            self.runtime.profile_callsite(stacklevel + 1, False)

    def get_scalar_array(self, stacklevel):
        if self.deferred is not None:
            return self.deferred.get_scalar_array(stacklevel=(stacklevel + 1))
        return self.array.reshape(())

    def _create_indexing_key(self, key, stacklevel):
        if key is None or key is Ellipsis:
            return key
        if isinstance(key, int):
            return key
        if isinstance(key, slice):
            return key
        if isinstance(key, tuple):
            result = ()
            for k in key:
                result += (self._create_indexing_key(k, stacklevel + 1),)
            return result
        assert isinstance(key, NumPyThunk)
        return self.runtime.to_eager_array(
            key, stacklevel=(stacklevel + 1)
        ).array

    def get_item(self, key, stacklevel, view=None, dim_map=None):
        if self.deferred is not None:
            return self.deferred.get_item(
                key, stacklevel=(stacklevel + 1), view=view, dim_map=dim_map
            )
        if self._is_advanced_indexing(key):
            index_key = self._create_indexing_key(key, stacklevel + 1)
            out = self.array[index_key]
            result = EagerArray(self.runtime, out)
        else:
            child = self.array[key]
            result = EagerArray(self.runtime, child, parent=self, key=key)
            if self.children is None:
                self.children = list()
            self.children.append(result)
        return result

    def set_item(self, key, value, stacklevel):
        if self.shadow:
            value = self.runtime.to_eager_array(
                value, stacklevel=(stacklevel + 1)
            )
        elif self.deferred is None:
            self.check_eager_args((stacklevel + 1), value)
        if self.deferred is not None:
            self.deferred.set_item(key, value, stacklevel=(stacklevel + 1))
        else:
            if self._is_advanced_indexing(key):
                index_key = self._create_indexing_key(key, stacklevel + 1)
                if isinstance(value, EagerArray):
                    self.array[index_key] = value.array
                else:
                    self.array[index_key] = value
            else:
                if isinstance(value, EagerArray):
                    self.array[key] = value.array
                else:
                    self.array[key] = value

    def reshape(self, newshape, order, stacklevel):
        if self.deferred is not None:
            return self.deferred.reshape(
                newshape, order, stacklevel=(stacklevel + 1)
            )
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

    def squeeze(self, axis, stacklevel):
        if self.deferred is not None:
            return self.deferred.squeeze(axis, stacklevel=(stacklevel + 1))
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

    def swapaxes(self, axis1, axis2, stacklevel):
        if self.deferred is not None:
            return self.deferred.swapaxes(
                axis1, axis2, stacklevel=(stacklevel + 1)
            )
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

    def convert(self, rhs, stacklevel, warn=True):
        if self.shadow:
            rhs = self.runtime.to_eager_array(rhs, stacklevel=(stacklevel + 1))
        elif self.deferred is None:
            self.check_eager_args((stacklevel + 1), rhs)
        if self.deferred is not None:
            return self.deferred.convert(
                rhs, stacklevel=(stacklevel + 1), warn=warn
            )
        else:
            if self.array.size == 1:
                self.array.fill(rhs.array.item())
            else:
                self.array[:] = rhs.array
            self.runtime.profile_callsite(stacklevel + 1, False)

    def fill(self, value, stacklevel):
        if self.deferred is not None:
            self.deferred.fill(value, stacklevel=(stacklevel + 1))
        else:
            self.array.fill(value)
            self.runtime.profile_callsite(stacklevel + 1, False)

    def dot(self, rhs1, rhs2, stacklevel):
        if self.shadow:
            rhs1 = self.runtime.to_eager_array(
                rhs1, stacklevel=(stacklevel + 1)
            )
            rhs2 = self.runtime.to_eager_array(
                rhs2, stacklevel=(stacklevel + 1)
            )
        elif self.deferred is None:
            self.check_eager_args((stacklevel + 1), rhs1, rhs2)
        if self.deferred is not None:
            self.deferred.dot(rhs1, rhs2, stacklevel=(stacklevel + 1))
        else:
            np.dot(rhs1.array, rhs2.array, out=self.array)
            self.runtime.profile_callsite(stacklevel + 1, False)

    def transpose(self, rhs, axes, stacklevel):
        if self.shadow:
            rhs = self.runtime.to_eager_array(rhs, stacklevel=(stacklevel + 1))
        elif self.deferred is None:
            self.check_eager_args((stacklevel + 1), rhs)
        if self.deferred is not None:
            self.deferred.transpose(rhs, axes, stacklevel=(stacklevel + 1))
        else:
            if self.array.size == 1:
                self.array.fill(rhs.array.item())
            else:
                self.array[:] = np.transpose(rhs.array, axes)
            self.runtime.profile_callsite(stacklevel + 1, False)

    def diag(self, rhs, extract, k, stacklevel):
        if self.shadow:
            rhs = self.runtime.to_eager_array(rhs, stacklevel=(stacklevel + 1))
        elif self.deferred is None:
            self.check_eager_args((stacklevel + 1), rhs)
        if self.deferred is not None:
            self.deferred.diag(rhs, extract, k, stacklevel=(stacklevel + 1))
        else:
            self.array[:] = np.diag(rhs.array, k)
            self.runtime.profile_callsite(stacklevel + 1, False)

    def eye(self, k, stacklevel):
        if self.deferred is not None:
            self.deferred.eye(k, stacklevel=(stacklevel + 1))
        else:
            if self.array.size == 1:
                self.array.fill(1)
            else:
                self.array[:] = np.eye(
                    self.shape[0], self.shape[1], k, dtype=self.dtype
                )
            self.runtime.profile_callsite(stacklevel + 1, False)

    def arange(self, start, stop, step, stacklevel):
        if self.deferred is not None:
            self.deferred.arange(
                start, stop, step, stacklevel=(stacklevel + 1)
            )
        else:
            self.array = np.arange(start, stop, step, self.dtype)
            self.runtime.profile_callsite(stacklevel + 1, False)

    def tile(self, rhs, reps, stacklevel):
        if self.shadow:
            rhs = self.runtime.to_eager_array(rhs, stacklevel=(stacklevel + 1))
        elif self.deferred is None:
            self.check_eager_args((stacklevel + 1), rhs)
        if self.deferred is not None:
            self.deferred.tile(rhs, reps, stacklevel=(stacklevel + 1))
        else:
            self.array[:] = np.tile(rhs.array, reps)
            self.runtime.profile_callsite(stacklevel + 1, False)

    def bincount(self, rhs, stacklevel, weights=None):
        if self.shadow:
            rhs = self.runtime.to_eager_array(rhs, stacklevel=(stacklevel + 1))
            if weights is not None:
                weights = self.runtime.to_eager_array(
                    weights, stacklevel=(stacklevel + 1)
                )
        elif self.deferred is None:
            if weights is not None:
                self.check_eager_args((stacklevel + 1), rhs, weights)
            else:
                self.check_eager_args((stacklevel + 1), rhs)
        if self.deferred is not None:
            self.deferred.bincount(
                rhs, stacklevel=(stacklevel + 1), weights=weights
            )
        else:
            self.array[:] = np.bincount(
                rhs.array,
                weights.array if weights is not None else None,
                minlength=self.array.size,
            )
            self.runtime.profile_callsite(stacklevel + 1, False)

    def count_nonzero(self, axis, stacklevel):
        if self.deferred is not None:
            return self.deferred.count_nonzero(stacklevel=(stacklevel + 1))
        else:
            result = np.count_nonzero(self.array, axis)
            if isinstance(result, np.ndarray):
                return EagerArray(self.runtime, result)
            else:
                return result

    def nonzero(self, stacklevel):
        if self.deferred is not None:
            return self.deferred.nonzero(stacklevel=(stacklevel + 1))
        else:
            arrays = self.array.nonzero()
            result = ()
            for array in arrays:
                result += (EagerArray(self.runtime, array),)
            return result

    def sort(self, rhs, stacklevel):
        if self.shadow:
            rhs = self.runtime.to_eager_array(rhs, stacklevel=(stacklevel + 1))
        elif self.deferred is None:
            self.check_eager_args((stacklevel + 1), rhs)
        if self.deferred is not None:
            self.deferred.sort(rhs, stacklevel=(stacklevel + 1))
        else:
            self.array[:] = np.sort(rhs.array)
            self.runtime.profile_callsite(stacklevel + 1, False)

    def random_uniform(self, stacklevel):
        assert not self.shadow
        if self.deferred is not None:
            self.deferred.random_uniform(stacklevel=(stacklevel + 1))
        else:
            if self.array.size == 1:
                self.array.fill(np.random.rand())
            else:
                self.array[:] = np.random.rand(*(self.array.shape))
            self.runtime.profile_callsite(stacklevel + 1, False)

    def random_normal(self, stacklevel):
        assert not self.shadow
        if self.deferred is not None:
            self.deferred.random_normal(stacklevel=(stacklevel + 1))
        else:
            if self.array.size == 1:
                self.array.fill(np.random.randn())
            else:
                self.array[:] = np.random.randn(*(self.array.shape))
            self.runtime.profile_callsite(stacklevel + 1, False)

    def random_integer(self, low, high, stacklevel):
        assert not self.shadow
        if self.deferred is not None:
            self.deferred.random_integer(stacklevel=(stacklevel + 1))
        else:
            if self.array.size == 1:
                self.array.fill(np.random.randint(low, high))
            else:
                self.array[:] = np.random.randint(
                    low, high, size=self.array.shape, dtype=self.array.dtype
                )
            self.runtime.profile_callsite(stacklevel + 1, False)

    def unary_op(self, op, op_type, rhs, where, args, stacklevel):
        if self.shadow:
            rhs = self.runtime.to_eager_array(rhs, stacklevel=(stacklevel + 1))
            if where is not None and isinstance(where, NumPyThunk):
                where = self.runtime.to_eager_array(
                    where, stacklevel=(stacklevel + 1)
                )
        elif self.deferred is None:
            if where is not None and isinstance(where, NumPyThunk):
                self.check_eager_args((stacklevel + 1), rhs, where)
            else:
                self.check_eager_args((stacklevel + 1), rhs)
        if self.deferred is not None:
            self.deferred.unary_op(
                op, op_type, rhs, where, args, stacklevel=(stacklevel + 1)
            )
            return
        if op == NumPyOpCode.ABSOLUTE:
            np.absolute(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.ARCCOS:
            np.arccos(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.ARCSIN:
            np.arcsin(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.ARCTAN:
            np.arctan(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.CEIL:
            np.ceil(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.CLIP:
            np.clip(rhs.array, out=self.array, a_min=args[0], a_max=args[1])
        elif op == NumPyOpCode.COS:
            np.cos(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.EXP:
            np.exp(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.FLOOR:
            np.floor(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.INVERT:
            np.invert(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.ISINF:
            np.isinf(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.ISNAN:
            np.isnan(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.LOG:
            np.log(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.LOGICAL_NOT:
            np.logical_not(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.NEGATIVE:
            np.negative(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.SIN:
            np.sin(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.SQRT:
            np.sqrt(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.TAN:
            np.tan(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == NumPyOpCode.TANH:
            np.tanh(
                rhs.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        else:
            raise RuntimeError("unsupported unary op " + str(op))
        self.runtime.profile_callsite(stacklevel + 1, False)

    def unary_reduction(
        self, op, redop, rhs, where, axes, keepdims, args, initial, stacklevel
    ):
        if self.shadow:
            rhs = self.runtime.to_eager_array(rhs, stacklevel=(stacklevel + 1))
            if isinstance(where, NumPyThunk):
                where = self.runtime.to_eager_array(
                    where, stacklevel=(stacklevel + 1)
                )
        elif self.deferred is None:
            self.check_eager_args((stacklevel + 1), rhs)
        if self.deferred is not None:
            self.deferred.unary_reduction(
                op,
                redop,
                rhs,
                where,
                axes,
                keepdims,
                args,
                initial,
                stacklevel=(stacklevel + 1),
            )
            return
        if op == NumPyOpCode.ARGMAX:
            assert len(axes) == 1
            np.argmax(rhs.array, out=self.array, axis=axes[0])
        elif op == NumPyOpCode.ARGMIN:
            assert len(axes) == 1
            np.argmin(rhs.array, out=self.array, axis=axes[0])
        elif op == NumPyOpCode.CONTAINS:
            self.array.fill(args[0] in rhs.array)
        elif op == NumPyOpCode.MAX:
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
        elif op == NumPyOpCode.MIN:
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
        elif op == NumPyOpCode.NORM:
            if self.array.size == 1:
                self.array.fill(
                    np.linalg.norm(
                        rhs.array,
                        axis=axes,
                        keepdims=keepdims,
                        ord=args[0].item(),
                    )
                )
            else:
                self.array[:] = np.linalg.norm(
                    rhs.array, axis=axes, keepdims=keepdims, ord=args[0].item()
                )
        elif op == NumPyOpCode.PROD:
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
        elif op == NumPyOpCode.SUM:
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
        elif op == NumPyOpCode.COUNT_NONZERO:
            self.array[()] = np.count_nonzero(rhs.array, axis=axes)
        else:
            raise RuntimeError("unsupported unary reduction op " + str(op))
        self.runtime.profile_callsite(stacklevel + 1, False)

    def binary_op(self, op, rhs1, rhs2, where, args, stacklevel):
        if self.shadow:
            rhs1 = self.runtime.to_eager_array(
                rhs1, stacklevel=(stacklevel + 1)
            )
            rhs2 = self.runtime.to_eager_array(
                rhs2, stacklevel=(stacklevel + 1)
            )
            if where is not None and isinstance(where, NumPyThunk):
                where = self.runtime.to_eager_array(
                    where, stacklevel=(stacklevel + 1)
                )
        elif self.deferred is None:
            if where is not None and isinstance(where, NumPyThunk):
                self.check_eager_args((stacklevel + 1), rhs1, rhs2, where)
            else:
                self.check_eager_args((stacklevel + 1), rhs1, rhs2)
        if self.deferred is not None:
            self.deferred.binary_op(
                op, rhs1, rhs2, where, args, stacklevel=(stacklevel + 1)
            )
        else:
            if op == NumPyOpCode.ADD:
                np.add(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.LOGICAL_AND:
                np.logical_and(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.DIVIDE:
                np.divide(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.EQUAL:
                np.equal(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.FLOOR_DIVIDE:
                np.floor_divide(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.GREATER_EQUAL:
                np.greater_equal(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.GREATER:
                np.greater(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            # elif op == NumPyOpCode.SHIFT_LEFT:
            #    np.left_shift(rhs1.array, rhs2.array, out=self.array,
            #            where=where if not isinstance(where, EagerArray)
            #                        else where.array)
            # elif op == NumPyOpCode.SHIFT_RIGHT:
            #    np.right_shift(rhs1.array, rhs2.array, out=self.array,
            #            where=where if not isinstance(where, EagerArray)
            #                        else where.array)
            elif op == NumPyOpCode.MOD:
                np.mod(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.MULTIPLY:
                np.multiply(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.LOGICAL_OR:
                np.logical_or(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.POWER:
                np.power(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.SUBTRACT:
                np.subtract(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.LOGICAL_XOR:
                np.logical_xor(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.LESS_EQUAL:
                np.less_equal(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.LESS:
                np.less(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.MAXIMUM:
                np.maximum(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.MINIMUM:
                np.minimum(
                    rhs1.array,
                    rhs2.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            elif op == NumPyOpCode.NOT_EQUAL:
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
            self.runtime.profile_callsite(stacklevel + 1, False)

    def binary_reduction(self, op, rhs1, rhs2, broadcast, args, stacklevel):
        if self.shadow:
            rhs1 = self.runtime.to_eager_array(
                rhs1, stacklevel=(stacklevel + 1)
            )
            rhs2 = self.runtime.to_eager_array(
                rhs2, stacklevel=(stacklevel + 1)
            )
        elif self.deferred is None:
            self.check_eager_args((stacklevel + 1), rhs1, rhs2)
        if self.deferred is not None:
            self.deferred.binary_reduction(
                op, rhs1, rhs2, broadcast, args, stacklevel=(stacklevel + 1)
            )
        else:
            if op == NumPyOpCode.ALLCLOSE:
                self.array = np.allclose(
                    rhs1.array, rhs2.array, rtol=args[0], atol=args[1]
                )
            elif op == NumPyOpCode.EQUAL:
                self.array = np.array_equal(rhs1.array, rhs2.array)
            else:
                raise RuntimeError(
                    "unsupported binary reduction op " + str(op)
                )
            self.runtime.profile_callsite(stacklevel + 1, False)

    def ternary_op(self, op, rhs1, rhs2, rhs3, where, args, stacklevel):
        if self.shadow:
            rhs1 = self.runtime.to_eager_array(
                rhs1, stacklevel=(stacklevel + 1)
            )
            rhs2 = self.runtime.to_eager_array(
                rhs2, stacklevel=(stacklevel + 1)
            )
            rhs3 = self.runtime.to_eager_array(
                rhs3, stacklevel=(stacklevel + 1)
            )
            if where is not None and isinstance(where, NumPyThunk):
                where = self.runtime.to_eager_array(
                    where, stacklevel=(stacklevel + 1)
                )
        elif self.deferred is None:
            if where is not None and isinstance(where, NumPyThunk):
                self.check_eager_args(
                    (stacklevel + 1), rhs1, rhs2, rhs3, where
                )
            else:
                self.check_eager_args((stacklevel + 1), rhs1, rhs2, rhs3)
        if self.deferred is not None:
            self.deferred.ternary_op(
                op, rhs1, rhs2, rhs3, where, args, stacklevel=(stacklevel + 1)
            )
        else:
            if op == NumPyOpCode.WHERE:
                self.array[:] = np.where(rhs1.array, rhs2.array, rhs3.array)
            else:
                raise RuntimeError(
                    "unsupported ternary reduction op " + str(op)
                )
            self.runtime.profile_callsite(stacklevel + 1, False)
