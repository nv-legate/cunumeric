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

from typing import TYPE_CHECKING, Any

import numpy as np

from .config import (
    FFT_C2R,
    FFT_D2Z,
    FFT_R2C,
    FFT_Z2D,
    BinaryOpCode,
    FFTDirection,
    UnaryOpCode,
    UnaryRedCode,
    WindowOpCode,
)
from .thunk import NumPyThunk
from .utils import is_advanced_indexing

if TYPE_CHECKING:
    import numpy.typing as npt

    from .deferred import DeferredArray


_UNARY_OPS = {
    UnaryOpCode.ABSOLUTE: np.absolute,
    UnaryOpCode.ARCCOS: np.arccos,
    UnaryOpCode.ARCCOSH: np.arccosh,
    UnaryOpCode.ARCSIN: np.arcsin,
    UnaryOpCode.ARCSINH: np.arcsinh,
    UnaryOpCode.ARCTAN: np.arctan,
    UnaryOpCode.ARCTANH: np.arctanh,
    UnaryOpCode.CBRT: np.cbrt,
    UnaryOpCode.CEIL: np.ceil,
    UnaryOpCode.CONJ: np.conj,
    UnaryOpCode.COS: np.cos,
    UnaryOpCode.COSH: np.cosh,
    UnaryOpCode.DEG2RAD: np.deg2rad,
    UnaryOpCode.EXP2: np.exp2,
    UnaryOpCode.EXP: np.exp,
    UnaryOpCode.EXPM1: np.expm1,
    UnaryOpCode.FLOOR: np.floor,
    UnaryOpCode.FREXP: np.frexp,
    UnaryOpCode.INVERT: np.invert,
    UnaryOpCode.ISFINITE: np.isfinite,
    UnaryOpCode.ISINF: np.isinf,
    UnaryOpCode.ISNAN: np.isnan,
    UnaryOpCode.LOG10: np.log10,
    UnaryOpCode.LOG1P: np.log1p,
    UnaryOpCode.LOG2: np.log2,
    UnaryOpCode.LOG: np.log,
    UnaryOpCode.LOGICAL_NOT: np.logical_not,
    UnaryOpCode.MODF: np.modf,
    UnaryOpCode.NEGATIVE: np.negative,
    UnaryOpCode.POSITIVE: np.positive,
    UnaryOpCode.RAD2DEG: np.rad2deg,
    UnaryOpCode.RECIPROCAL: np.reciprocal,
    UnaryOpCode.RINT: np.rint,
    UnaryOpCode.SIGN: np.sign,
    UnaryOpCode.SIGNBIT: np.signbit,
    UnaryOpCode.SIN: np.sin,
    UnaryOpCode.SINH: np.sinh,
    UnaryOpCode.SQRT: np.sqrt,
    UnaryOpCode.SQUARE: np.square,
    UnaryOpCode.TAN: np.tan,
    UnaryOpCode.TANH: np.tanh,
    UnaryOpCode.TRUNC: np.trunc,
}

_UNARY_RED_OPS = {
    UnaryRedCode.ALL: np.all,
    UnaryRedCode.ANY: np.any,
    UnaryRedCode.MAX: np.max,
    UnaryRedCode.MIN: np.min,
    UnaryRedCode.PROD: np.prod,
    UnaryRedCode.SUM: np.sum,
}

_BINARY_OPS = {
    BinaryOpCode.ADD: np.add,
    BinaryOpCode.ARCTAN2: np.arctan2,
    BinaryOpCode.BITWISE_AND: np.bitwise_and,
    BinaryOpCode.BITWISE_OR: np.bitwise_or,
    BinaryOpCode.BITWISE_XOR: np.bitwise_xor,
    BinaryOpCode.COPYSIGN: np.copysign,
    BinaryOpCode.DIVIDE: np.divide,
    BinaryOpCode.EQUAL: np.equal,
    BinaryOpCode.FLOAT_POWER: np.float_power,
    BinaryOpCode.FLOOR_DIVIDE: np.floor_divide,
    BinaryOpCode.FMOD: np.fmod,
    BinaryOpCode.GCD: np.gcd,
    BinaryOpCode.GREATER: np.greater,
    BinaryOpCode.GREATER_EQUAL: np.greater_equal,
    BinaryOpCode.HYPOT: np.hypot,
    BinaryOpCode.LCM: np.lcm,
    BinaryOpCode.LDEXP: np.ldexp,
    BinaryOpCode.LEFT_SHIFT: np.left_shift,
    BinaryOpCode.LESS: np.less,
    BinaryOpCode.LESS_EQUAL: np.less_equal,
    BinaryOpCode.LOGADDEXP2: np.logaddexp2,
    BinaryOpCode.LOGADDEXP: np.logaddexp,
    BinaryOpCode.LOGICAL_AND: np.logical_and,
    BinaryOpCode.LOGICAL_OR: np.logical_or,
    BinaryOpCode.LOGICAL_XOR: np.logical_xor,
    BinaryOpCode.MAXIMUM: np.maximum,
    BinaryOpCode.MINIMUM: np.minimum,
    BinaryOpCode.MOD: np.mod,
    BinaryOpCode.MULTIPLY: np.multiply,
    BinaryOpCode.NEXTAFTER: np.nextafter,
    BinaryOpCode.NOT_EQUAL: np.not_equal,
    BinaryOpCode.POWER: np.power,
    BinaryOpCode.RIGHT_SHIFT: np.right_shift,
    BinaryOpCode.SUBTRACT: np.subtract,
}

_WINDOW_OPS = {
    WindowOpCode.BARLETT: np.bartlett,
    WindowOpCode.BLACKMAN: np.blackman,
    WindowOpCode.HAMMING: np.hamming,
    WindowOpCode.HANNING: np.hanning,
    WindowOpCode.KAISER: np.kaiser,
}


def eye_reference(shape, dtype, axes):
    n = min(shape[ax] for ax in axes)
    res = np.zeros(shape, dtype=dtype)
    for i in range(n):
        sl = tuple(
            i if ax in axes else slice(None) for ax in range(len(shape))
        )
        res[sl] = 1
    return res


def diagonal_reference(a, axes):
    transpose_axes = tuple(ax for ax in range(a.ndim) if ax not in axes)
    axes = sorted(axes, reverse=False, key=lambda i: a.shape[i])
    axes = tuple(axes)
    a = a.transpose(transpose_axes + axes)
    diff = a.ndim - len(axes)
    axes = tuple((diff + ax) for ax in range(0, len(axes)))
    eye = eye_reference(a.shape, a.dtype, axes)
    res = a * eye
    for ax in list(reversed(sorted(axes)))[:-1]:
        res = res.sum(axis=ax)
    return res


class EagerArray(NumPyThunk):
    """This is an eager thunk for describing NumPy computations.
    It is backed by a standard NumPy array that stores the result
    of the computation locally.

    :meta private:
    """

    def __init__(self, runtime, array, parent=None, key=None) -> None:
        super().__init__(runtime, array.dtype)
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

    def __numpy_array__(self) -> npt.NDArray[Any]:
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
        if self.deferred is not None:
            return
        for arg in args:
            if self.runtime.is_eager_array(arg):
                if arg.deferred is not None:
                    self.to_deferred_array()
                    break
            elif self.runtime.is_deferred_array(arg):
                self.to_deferred_array()
                break
            elif arg is None or not isinstance(arg, NumPyThunk):
                pass
            else:
                raise RuntimeError("bad argument type")

    def _convert_children(self):
        """
        Traverse down our children and convert them to deferred arrays.
        """
        assert self.runtime.is_deferred_array(self.deferred)
        if self.children is not None:
            for child in self.children:
                if child.deferred is None:
                    func = getattr(self.deferred, child.key[0])
                    args = child.key[1:]
                    child.deferred = func(*args)
            # After we've made all the deferred views for each child then
            # we can traverse down. Do it this way so we can get partition
            # coalescing where possible
            for child in self.children:
                child._convert_children()

    def to_deferred_array(self) -> DeferredArray:
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
                    self.deferred = self.runtime.create_wrapped_scalar(
                        self.array.data,
                        dtype=self.array.dtype,
                        shape=self.shape,
                    )
                else:
                    self.deferred = self.runtime.find_or_create_array_thunk(
                        self.array,
                        share=self.escaped,
                        defer=True,
                    )
                self._convert_children()
            else:
                # Traverse up the tree to make the deferred array
                self.parent.to_deferred_array()
                assert self.deferred is not None
        return self.deferred

    def imag(self):
        if self.deferred is not None:
            return self.deferred.imag()
        return EagerArray(self.runtime, self.array.imag)

    def real(self):
        if self.deferred is not None:
            return self.deferred.real()
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

    def fft(self, rhs, axes, kind, direction):
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.fft(rhs, axes, kind, direction)
        else:
            if kind in (FFT_D2Z, FFT_R2C):
                res = np.fft.rfftn(rhs.array, axes=axes, norm="backward")
            elif kind in (FFT_Z2D, FFT_C2R):
                s = tuple(rhs.array.shape[i] for i in axes)
                res = np.fft.irfftn(rhs.array, s=s, axes=axes, norm="forward")
            else:
                if direction == FFTDirection.FORWARD:
                    res = np.fft.fftn(rhs.array, axes=axes, norm="backward")
                else:
                    res = np.fft.ifftn(rhs.array, axes=axes, norm="forward")
            if kind.is_single_precision:
                if res.dtype == np.complex128:
                    self.array[:] = res.astype(np.complex64)
                elif res.dtype == np.float64:
                    self.array[:] = res.astype(np.float32)
                else:
                    raise RuntimeError("Unsupported data type in eager FFT")
            else:
                self.array[:] = res

    def copy(self, rhs, deep=False):
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

    def get_item(self, key) -> NumPyThunk:
        if self.deferred is not None:
            return self.deferred.get_item(key)
        if is_advanced_indexing(key):
            index_key = self._create_indexing_key(key)
            out = self.array[index_key]
            result = EagerArray(self.runtime, out)
        else:
            child = self.array[key]
            result = EagerArray(
                self.runtime, child, parent=self, key=("get_item", key)
            )
            if self.children is None:
                self.children = list()
            self.children.append(result)
        return result

    def set_item(self, key, value):
        self.check_eager_args(value)
        if self.deferred is not None:
            self.deferred.set_item(key, value)
        else:
            if is_advanced_indexing(key):
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
        if child.base is None:
            result = EagerArray(self.runtime, child)
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
        # Early exit if there's no dimension to squeeze
        if child is self.array:
            return self
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
        self.check_eager_args(rhs)
        if self.deferred is not None:
            return self.deferred.convert(rhs, warn=warn)
        else:
            if self.array.size == 1:
                self.array.fill(rhs.array.item())
            else:
                if (
                    rhs.array.dtype.kind == "c"
                    and self.array.dtype.kind != "c"
                ):
                    self.array[:] = rhs.array.real
                else:
                    self.array[:] = rhs.array

    def fill(self, value) -> None:
        if self.deferred is not None:
            self.deferred.fill(value)
        else:
            self.array.fill(value)

    def dot(self, rhs1, rhs2):
        self.check_eager_args(rhs1, rhs2)
        if self.deferred is not None:
            self.deferred.dot(rhs1, rhs2)
        else:
            np.dot(rhs1.array, rhs2.array, out=self.array)

    def transpose(self, axes):
        if self.deferred is not None:
            return self.deferred.transpose(axes)
        child = self.array.transpose(axes)
        # Should be aliased with parent region
        assert child.base is not None
        result = EagerArray(
            self.runtime, child, parent=self, key=("transpose", axes)
        )
        if self.children is None:
            self.children = list()
        self.children.append(result)
        return result

    def repeat(self, repeats, axis, scalar_repeats) -> NumPyThunk:
        if not scalar_repeats:
            self.check_eager_args(repeats)
        if self.deferred is not None:
            return self.deferred.repeat(
                repeats,
                axis,
                scalar_repeats,
            )
        else:
            if not scalar_repeats:
                array = np.repeat(self.array, repeats.array, axis)
            else:
                array = np.repeat(self.array, repeats, axis)
            return EagerArray(self.runtime, array)

    def flip(self, rhs, axes):
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.flip(rhs, axes)
        else:
            self.array = np.flip(rhs.array, axes)

    def broadcast_to(self, shape):
        if self.deferred is not None:
            return self.deferred.broadcast_to(shape)
        else:
            return EagerArray(self.runtime, np.broadcast_to(self.array, shape))

    def contract(
        self,
        lhs_modes,
        rhs1_thunk,
        rhs1_modes,
        rhs2_thunk,
        rhs2_modes,
        mode2extent,
    ):
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

    def choose(self, rhs, *args):
        self.check_eager_args(*args, rhs)
        if self.deferred is not None:
            self.deferred.choose(
                rhs,
                *args,
            )
        else:
            choices = tuple(c.array for c in args)
            self.array[:] = np.choose(rhs.array, choices, mode="raise")

    def _diag_helper(self, rhs, offset, naxes, extract, trace):
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred._diag_helper(rhs, offset, naxes, extract, trace)
        else:
            if (naxes == 2) and extract and not trace:
                ndims = rhs.array.ndim
                self.array[:] = np.diagonal(
                    rhs.array, offset=offset, axis1=ndims - 2, axis2=ndims - 1
                )
            elif (naxes < 2) and not extract:
                self.array[:] = np.diag(rhs.array, offset)
            elif (naxes >= 2) and trace:
                ndim = rhs.array.ndim
                self.array[:] = np.trace(
                    rhs.array, offset=offset, axis1=ndim - 2, axis2=ndim - 1
                )
            else:  # naxes>2
                ndims = rhs.array.ndim
                axes = tuple(range(ndims - naxes, ndims))
                self.array = diagonal_reference(rhs.array, axes)

    def eye(self, k) -> None:
        if self.deferred is not None:
            self.deferred.eye(k)
        else:
            if self.array.size == 1:
                self.array.fill(1)
            else:
                self.array[:] = np.eye(
                    self.shape[0], self.shape[1], k, dtype=self.dtype
                )

    def arange(self, start, stop, step) -> None:
        if self.deferred is not None:
            self.deferred.arange(start, stop, step)
        else:
            self.array = np.arange(start, stop, step, self.dtype)

    def tile(self, rhs, reps):
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.tile(rhs, reps)
        else:
            self.array[:] = np.tile(rhs.array, reps)

    def bincount(self, rhs, weights=None):
        self.check_eager_args(rhs, weights)
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

    def searchsorted(self, rhs, v, side="left"):
        self.check_eager_args(rhs, v)
        if self.deferred is not None:
            self.deferred.searchsorted(rhs, v, side)
        else:
            self.array = np.searchsorted(rhs.array, v.array, side=side)

    def sort(self, rhs, argsort=False, axis=-1, kind="quicksort", order=None):
        self.check_eager_args(rhs, axis, kind, order)
        if self.deferred is not None:
            self.deferred.sort(rhs, argsort, axis, kind, order)
        else:
            if argsort:
                self.array = np.argsort(rhs.array, axis, kind, order)
            else:
                self.array = np.sort(rhs.array, axis, kind, order)

    def partition(
        self,
        rhs,
        kth,
        argpartition=False,
        axis=-1,
        kind="introselect",
        order=None,
    ):
        self.check_eager_args(rhs, kth, axis, kind, order)
        if self.deferred is not None:
            self.deferred.partition(rhs, kth, argpartition, axis, kind, order)
        else:
            if argpartition:
                self.array = np.argpartition(rhs.array, kth, axis, kind, order)
            else:
                self.array = np.partition(rhs.array, kth, axis, kind, order)

    def random_uniform(self) -> None:
        if self.deferred is not None:
            self.deferred.random_uniform()
        else:
            if self.array.size == 1:
                self.array.fill(np.random.rand())
            else:
                self.array[:] = np.random.rand(*(self.array.shape))

    def random_normal(self) -> None:
        if self.deferred is not None:
            self.deferred.random_normal()
        else:
            if self.array.size == 1:
                self.array.fill(np.random.randn())
            else:
                self.array[:] = np.random.randn(*(self.array.shape))

    def random_integer(self, low, high) -> None:
        if self.deferred is not None:
            self.deferred.random_integer(low, high)
        else:
            if self.array.size == 1:
                self.array.fill(np.random.randint(low, high))
            else:
                self.array[:] = np.random.randint(
                    low, high, size=self.array.shape, dtype=self.array.dtype
                )

    def unary_op(self, op, rhs, where, args, multiout=None):
        if multiout is None:
            self.check_eager_args(rhs, where)
        else:
            self.check_eager_args(rhs, where, *multiout)

        if self.deferred is not None:
            self.deferred.unary_op(op, rhs, where, args, multiout=multiout)
            return

        if op in _UNARY_OPS:
            func = _UNARY_OPS[op]
            if multiout is None:
                func(
                    rhs.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            else:
                func(
                    rhs.array,
                    out=(self.array, *(out.array for out in multiout)),
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
        elif op == UnaryOpCode.CLIP:
            np.clip(rhs.array, out=self.array, a_min=args[0], a_max=args[1])
        elif op == UnaryOpCode.COPY:
            self.array[:] = rhs.array[:]
        elif op == UnaryOpCode.IMAG:
            self.array = np.imag(rhs.array)
        elif op == UnaryOpCode.REAL:
            self.array = np.real(rhs.array)
        else:
            raise RuntimeError("unsupported unary op " + str(op))

    def unary_reduction(
        self, op, rhs, where, orig_axis, axes, keepdims, args, initial
    ):
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.unary_reduction(
                op,
                rhs,
                where,
                orig_axis,
                axes,
                keepdims,
                args,
                initial,
            )
            return
        if op in _UNARY_RED_OPS:
            fn = _UNARY_RED_OPS[op]
            if initial is None:
                # NumPy starts using this predefined constant, instead of None,
                # to mean no value was given by the caller
                initial = np._NoValue
            fn(
                rhs.array,
                out=self.array,
                axis=orig_axis,
                keepdims=keepdims,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )
        elif op == UnaryRedCode.ARGMAX:
            np.argmax(
                rhs.array, out=self.array, axis=orig_axis, keepdims=keepdims
            )
        elif op == UnaryRedCode.ARGMIN:
            np.argmin(
                rhs.array, out=self.array, axis=orig_axis, keepdims=keepdims
            )
        elif op == UnaryRedCode.CONTAINS:
            self.array.fill(args[0] in rhs.array)
        elif op == UnaryRedCode.COUNT_NONZERO:
            self.array[()] = np.count_nonzero(rhs.array, axis=orig_axis)
        else:
            raise RuntimeError("unsupported unary reduction op " + str(op))

    def isclose(self, rhs1, rhs2, rtol, atol, equal_nan) -> None:
        self.check_eager_args(rhs1, rhs2)
        if self.deferred is not None:
            self.deferred.isclose(rhs1, rhs2, rtol, atol, equal_nan)
        else:
            self.array[:] = np.isclose(
                rhs1.array,
                rhs2.array,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )

    def binary_op(self, op, rhs1, rhs2, where, args):
        self.check_eager_args(rhs1, rhs2, where)
        if self.deferred is not None:
            self.deferred.binary_op(op, rhs1, rhs2, where, args)
        else:
            func = _BINARY_OPS.get(op, None)
            if func is None:
                raise RuntimeError("unsupported binary op " + str(op))
            func(
                rhs1.array,
                rhs2.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )

    def binary_reduction(self, op, rhs1, rhs2, broadcast, args):
        self.check_eager_args(rhs1, rhs2)
        if self.deferred is not None:
            self.deferred.binary_reduction(op, rhs1, rhs2, broadcast, args)
        else:
            if op == BinaryOpCode.ISCLOSE:
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
        self.check_eager_args(rhs1, rhs2, rhs3)
        if self.deferred is not None:
            self.deferred.where(rhs1, rhs2, rhs3)
        else:
            self.array[:] = np.where(rhs1.array, rhs2.array, rhs3.array)

    def trilu(self, rhs, k, lower):
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.trilu(rhs, k, lower)
        else:
            if lower:
                self.array[:] = np.tril(rhs.array, k)
            else:
                self.array[:] = np.triu(rhs.array, k)

    def cholesky(self, src, no_tril):
        self.check_eager_args(src)
        if self.deferred is not None:
            self.deferred.cholesky(src, no_tril)
        else:
            try:
                result = np.linalg.cholesky(src.array)
            except np.linalg.LinAlgError as e:
                from .linalg import LinAlgError

                raise LinAlgError(e) from e
            if no_tril:
                result = np.triu(result.T.conj(), k=1) + result
            self.array[:] = result

    def unique(self):
        if self.deferred is not None:
            return self.deferred.unique()
        else:
            return EagerArray(self.runtime, np.unique(self.array))

    def create_window(self, op_code, M, *args) -> None:
        if self.deferred is not None:
            return self.deferred.create_window(op_code, M, *args)
        else:
            fn = _WINDOW_OPS[op_code]
            self.array[:] = fn(M, *args)

    def packbits(self, src, axis, bitorder):
        self.check_eager_args(src)
        if self.deferred is not None:
            self.deferred.packbits(src, axis, bitorder)
        else:
            self.array[:] = np.packbits(
                src.array, axis=axis, bitorder=bitorder
            )

    def unpackbits(self, src, axis, bitorder):
        self.check_eager_args(src)
        if self.deferred is not None:
            self.deferred.unpackbits(src, axis, bitorder)
        else:
            self.array[:] = np.unpackbits(
                src.array, axis=axis, bitorder=bitorder
            )
