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

from collections.abc import Iterable
from functools import reduce
from inspect import signature
from typing import Optional, Set, Tuple

import numpy as np
import pyarrow

from legate.core import Array

from .config import BinaryOpCode, UnaryOpCode, UnaryRedCode
from .runtime import runtime


def add_boilerplate(*array_params: str, mutates_self: bool = False):
    """
    Adds required boilerplate to the wrapped cuNumeric ndarray member function.

    Every time the wrapped function is called, this wrapper will:
    * Convert all specified array-like parameters, plus the special "out"
      parameter (if present), to cuNumeric ndarrays.
    * Convert the special "where" parameter (if present) to a valid predicate.
    * Handle the case of scalar cuNumeric ndarrays, by forwarding the operation
      to the equivalent `()`-shape numpy array.

    NOTE: Assumes that no parameters are mutated besides `out`, and `self` if
    `mutates_self` is True.
    """
    keys: Set[str] = set(array_params)

    def decorator(func):
        assert not hasattr(
            func, "__wrapped__"
        ), "this decorator must be the innermost"

        # For each parameter specified by name, also consider the case where
        # it's passed as a positional parameter.
        indices: Set[int] = set()
        all_formals: Set[str] = set()
        where_idx: Optional[int] = None
        out_idx: Optional[int] = None
        for (idx, param) in enumerate(signature(func).parameters):
            all_formals.add(param)
            if param == "where":
                where_idx = idx
            elif param == "out":
                assert not mutates_self
                out_idx = idx
            elif param in keys:
                indices.add(idx)
        assert len(keys - all_formals) == 0, "unkonwn parameter(s)"

        def wrapper(*args, **kwargs):
            self = args[0]
            assert (where_idx is None or len(args) <= where_idx) and (
                out_idx is None or len(args) <= out_idx
            ), "'where' and 'out' should be passed as keyword arguments"

            # Convert relevant arguments to cuNumeric ndarrays
            args = tuple(
                ndarray.convert_to_cunumeric_ndarray(arg)
                if idx in indices and arg is not None
                else arg
                for (idx, arg) in enumerate(args)
            )
            for (k, v) in kwargs.items():
                if v is None:
                    continue
                elif k == "where":
                    kwargs[k] = ndarray.convert_to_predicate_ndarray(v)
                elif k == "out":
                    kwargs[k] = ndarray.convert_to_cunumeric_ndarray(
                        v, share=True
                    )
                elif k in keys:
                    kwargs[k] = ndarray.convert_to_cunumeric_ndarray(v)

            # Handle the case where all array-like parameters are scalar, by
            # performing the operation on the equivalent scalar numpy arrays.
            # NOTE: This implicitly blocks on the contents of these arrays.
            if all(
                arg._thunk.scalar
                for (idx, arg) in enumerate(args)
                if (idx in indices or idx == 0) and isinstance(arg, ndarray)
            ) and all(
                v._thunk.scalar
                for (k, v) in kwargs.items()
                if (k in keys or k == "where") and isinstance(v, ndarray)
            ):
                out = None
                if "out" in kwargs:
                    out = kwargs["out"]
                    del kwargs["out"]
                args = tuple(
                    arg._thunk.__numpy_array__()
                    if (idx in indices or idx == 0)
                    and isinstance(arg, ndarray)
                    else arg
                    for (idx, arg) in enumerate(args)
                )
                for (k, v) in kwargs.items():
                    if (k in keys or k == "where") and isinstance(v, ndarray):
                        kwargs[k] = v._thunk.__numpy_array__()
                self_scalar = args[0]
                args = args[1:]
                res_scalar = getattr(self_scalar, func.__name__)(
                    *args, **kwargs
                )
                if mutates_self:
                    self._thunk = runtime.create_scalar(
                        self_scalar.data,
                        self_scalar.dtype,
                        shape=self_scalar.shape,
                        wrap=True,
                    )
                    return
                result = ndarray.convert_to_cunumeric_ndarray(res_scalar)
                if out is not None:
                    out._thunk.copy(result._thunk)
                    result = out
                return result

            return func(*args, **kwargs)

        return wrapper

    return decorator


def broadcast_shapes(*args):
    arrays = [np.empty(x, dtype=[]) for x in args]
    return np.broadcast(*arrays).shape


class ndarray(object):
    def __init__(
        self,
        shape,
        dtype=np.float64,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        thunk=None,
        inputs=None,
    ):
        if thunk is None:
            if not isinstance(dtype, np.dtype):
                dtype = np.dtype(dtype)
            if buffer is not None:
                # Make a normal numpy array for this buffer
                np_array = np.ndarray(
                    shape=shape,
                    dtype=dtype,
                    buffer=buffer,
                    offset=offset,
                    strides=strides,
                    order=order,
                )
                self._thunk = runtime.find_or_create_array_thunk(
                    np_array, share=False
                )
            else:
                # Filter the inputs if necessary
                if inputs is not None:
                    inputs = [
                        inp._thunk
                        for inp in inputs
                        if isinstance(inp, ndarray)
                    ]
                self._thunk = runtime.create_empty_thunk(shape, dtype, inputs)
        else:
            self._thunk = thunk
        self._legate_data = None

    # Support for the Legate data interface
    @property
    def __legate_data_interface__(self):
        if self._legate_data is None:
            # All of our thunks implement the Legate Store interface
            # so we just need to convert our type and stick it in
            # a Legate Array
            arrow_type = pyarrow.from_numpy_dtype(self.dtype)
            # We don't have nullable data for the moment
            # until we support masked arrays
            array = Array(arrow_type, [None, self._thunk])
            self._legate_data = dict()
            self._legate_data["version"] = 1
            data = dict()
            field = pyarrow.field(
                "cuNumeric Array", arrow_type, nullable=False
            )
            data[field] = array
            self._legate_data["data"] = data
        return self._legate_data

    # A class method for sanitizing inputs by converting them to
    # cuNumeric ndarray types
    @staticmethod
    def convert_to_cunumeric_ndarray(obj, share=False):
        # If this is an instance of one of our ndarrays then we're done
        if isinstance(obj, ndarray):
            return obj
        # Ask the runtime to make a numpy thunk for this object
        thunk = runtime.get_numpy_thunk(obj, share=share)
        return ndarray(shape=None, thunk=thunk)

    @staticmethod
    def convert_to_predicate_ndarray(obj):
        # Keep all boolean types as they are
        if obj is True or obj is False:
            return obj
        # GH #135
        raise NotImplementedError(
            "the `where` parameter is currently not supported"
        )

    # Properties for ndarray

    # Disable these since they seem to cause problems
    # when our arrays do not last long enough, instead
    # users will go through the __array__ method

    # @property
    # def __array_interface__(self):
    #    return self.__array__().__array_interface__

    # @property
    # def __array_priority__(self):
    #    return self.__array__().__array_priority__

    # @property
    # def __array_struct__(self):
    #    return self.__array__().__array_struct__

    @property
    def T(self):
        return self.transpose()

    @property
    def base(self):
        return self.__array__().base

    @property
    def data(self):
        return self.__array__().data

    @property
    def dtype(self):
        return self._thunk.dtype

    @property
    def flags(self):
        return self.__array__().flags

    @property
    def flat(self):
        return self.__array__().flat

    @property
    def imag(self):
        if self.dtype.kind == "c":
            return ndarray(shape=self.shape, thunk=self._thunk.imag())
        else:
            result = ndarray(self.shape, self.dtype)
            result.fill(0)
            return result

    @property
    def ndim(self):
        return self._thunk.ndim

    @property
    def real(self):
        if self.dtype.kind == "c":
            return ndarray(shape=self.shape, thunk=self._thunk.real())
        else:
            return self

    @property
    def shape(self):
        return self._thunk.shape

    @property
    def size(self):
        s = 1
        if self.ndim == 0:
            return s
        for p in self.shape:
            s *= p
        return s

    @property
    def itemsize(self):
        return self._thunk.dtype.itemsize

    @property
    def nbytes(self):
        return self.itemsize * self.size

    @property
    def strides(self):
        return self.__array__().strides

    @property
    def ctypes(self):
        return self.__array__().ctypes

    # Methods for ndarray

    def __abs__(self):
        # Handle the nice case of it being unsigned
        if (
            self.dtype.type == np.uint16
            or self.dtype.type == np.uint32
            or self.dtype.type == np.uint64
            or self.dtype.type == np.bool_
        ):
            return self
        return self.perform_unary_op(UnaryOpCode.ABSOLUTE, self)

    def __add__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.ADD, self, rhs_array)

    def __and__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.LOGICAL_AND, self, rhs_array
        )

    def __array__(self, dtype=None):
        if dtype is None:
            return self._thunk.__numpy_array__()
        else:
            return self._thunk.__numpy_array__().__array__(dtype)

    # def __array_prepare__(self, *args, **kwargs):
    #    return self.__array__().__array_prepare__(*args, **kwargs)

    # def __array_wrap__(self, *args, **kwargs):
    #    return self.__array__().__array_wrap__(*args, **kwargs)

    def __bool__(self):
        return bool(self.__array__())

    def __complex__(self):
        return complex(self.__array__())

    def __contains__(self, item):
        if isinstance(item, np.ndarray):
            args = (item.astype(self.dtype),)
        else:  # Otherwise convert it to a scalar numpy array of our type
            args = (np.array(item, dtype=self.dtype),)
        if args[0].size != 1:
            raise ValueError("contains needs scalar item")
        return self.perform_unary_reduction(
            UnaryRedCode.CONTAINS,
            self,
            axis=None,
            dtype=np.dtype(np.bool_),
            args=args,
            check_types=False,
        )

    def __copy__(self):
        result = ndarray(self.shape, self.dtype, inputs=(self,))
        result._thunk.copy(self._thunk, deep=False)
        return result

    def __deepcopy__(self, memo=None):
        result = ndarray(self.shape, self.dtype, inputs=(self,))
        result._thunk.copy(self._thunk, deep=True)
        return result

    def __div__(self, rhs):
        return self.internal_truediv(rhs, inplace=False)

    def __divmod__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.DIVMOD, self, rhs_array)

    def __eq__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.EQUAL, self, rhs_array, out_dtype=np.dtype(np.bool_)
        )

    def __float__(self):
        return float(self.__array__())

    def __floordiv__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.FLOOR_DIVIDE, self, rhs_array
        )

    def __format__(self, *args, **kwargs):
        return self.__array__().__format__(*args, **kwargs)

    def __ge__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.GREATER_EQUAL,
            self,
            rhs_array,
            out_dtype=np.dtype(np.bool_),
        )

    # __getattribute__

    def _convert_key(self, key, first=True):
        # Convert any arrays stored in a key to a cuNumeric array
        if (
            key is np.newaxis
            or key is Ellipsis
            or np.isscalar(key)
            or isinstance(key, slice)
        ):
            return (key,) if first else key
        elif isinstance(key, tuple) and first:
            return tuple(self._convert_key(k, first=False) for k in key)
        else:
            # Otherwise convert it to a cuNumeric array and get the thunk
            return self.convert_to_cunumeric_ndarray(key)._thunk

    @add_boilerplate()
    def __getitem__(self, key):
        key = self._convert_key(key)
        return ndarray(shape=None, thunk=self._thunk.get_item(key))

    def __gt__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.GREATER, self, rhs_array, out_dtype=np.dtype(np.bool_)
        )

    def __hash__(self, *args, **kwargs):
        raise TypeError("unhashable type: cunumeric.ndarray")

    def __iadd__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(BinaryOpCode.ADD, self, rhs_array, out=self)
        return self

    def __iand__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(
            BinaryOpCode.LOGICAL_AND, self, rhs_array, out=self
        )
        return self

    def __idiv__(self, rhs):
        return self.internal_truediv(rhs, inplace=True)

    def __idivmod__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(BinaryOpCode.DIVMOD, self, rhs_array, out=self)
        return self

    def __ifloordiv__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(
            BinaryOpCode.FLOOR_DIVIDE, self, rhs_array, out=self
        )
        return self

    def __ilshift__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(
            BinaryOpCode.SHIFT_LEFT, self, rhs_array, out=self
        )
        return self

    def __imod__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(BinaryOpCode.MODULUS, self, rhs_array, out=self)
        return self

    def __imul__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(
            BinaryOpCode.MULTIPLY, self, rhs_array, out=self
        )
        return self

    def __int__(self):
        return int(self.__array__())

    def __invert__(self):
        if self.dtype == np.bool_:
            # Boolean values are special, just do logical NOT
            return self.perform_unary_op(
                UnaryOpCode.LOGICAL_NOT, self, out_dtype=np.dtype(np.bool_)
            )
        else:
            return self.perform_unary_op(UnaryOpCode.INVERT, self)

    def __ior__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(
            BinaryOpCode.LOGICAL_OR, self, rhs_array, out=self
        )
        return self

    def __ipow__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(BinaryOpCode.POWER, self, rhs_array, out=self)
        return self

    def __irshift__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(
            BinaryOpCode.SHIFT_RIGHT, self, rhs_array, out=self
        )
        return self

    def __iter__(self):
        return self.__array__().__iter__()

    def __isub__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(
            BinaryOpCode.SUBTRACT, self, rhs_array, out=self
        )
        return self

    def internal_truediv(self, rhs, inplace):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self_array = self
        # Convert any non-floats to floating point arrays
        if self_array.dtype.kind != "f" and self_array.dtype.kind != "c":
            self_type = np.dtype(np.float64)
        else:
            self_type = self_array.dtype
        if rhs_array.dtype.kind != "f" and rhs_array.dtype.kind != "c":
            if inplace:
                rhs_type = self_type
            else:
                rhs_type = np.dtype(np.float64)
        else:
            rhs_type = rhs_array.dtype
        # If the types don't match then align them
        if self_type != rhs_type:
            common_type = self.find_common_type(self_array, rhs_array)
        else:
            common_type = self_type
        if self_array.dtype != common_type:
            temp = ndarray(
                self_array.shape,
                dtype=common_type,
                inputs=(self_array,),
            )
            temp._thunk.convert(self_array._thunk, warn=False)
            self_array = temp
        if rhs_array.dtype != common_type:
            temp = ndarray(
                rhs_array.shape,
                dtype=common_type,
                inputs=(rhs_array,),
            )
            temp._thunk.convert(rhs_array._thunk, warn=False)
            rhs_array = temp
        return self.perform_binary_op(
            BinaryOpCode.DIVIDE,
            self_array,
            rhs_array,
            out=self if inplace else None,
        )

    def __itruediv__(self, rhs):
        return self.internal_truediv(rhs, inplace=True)

    def __ixor__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        self.perform_binary_op(
            BinaryOpCode.LOGICAL_XOR, self, rhs_array, out=self
        )
        return self

    def __le__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.LESS_EQUAL,
            self,
            rhs_array,
            out_dtype=np.dtype(np.bool_),
        )

    def __len__(self):
        return self.shape[0]

    def __lshift__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.SHIFT_LEFT, self, rhs_array)

    def __lt__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.LESS, self, rhs_array, out_dtype=np.dtype(np.bool_)
        )

    def __matmul__(self, value):
        return self.dot(value)

    def __mod__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.MOD, self, rhs_array)

    def __mul__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.MULTIPLY, self, rhs_array)

    def __ne__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.NOT_EQUAL,
            self,
            rhs_array,
            out_dtype=np.dtype(np.bool_),
        )

    def __neg__(self):
        if (
            self.dtype.type == np.uint16
            or self.dtype.type == np.uint32
            or self.dtype.type == np.uint64
        ):
            raise TypeError("cannot negate unsigned type " + str(self.dtype))
        return self.perform_unary_op(UnaryOpCode.NEGATIVE, self)

    # __new__

    @add_boilerplate()
    def nonzero(self):
        thunks = self._thunk.nonzero()
        return tuple(
            ndarray(shape=thunk.shape, thunk=thunk) for thunk in thunks
        )

    def __nonzero__(self):
        return self.__array__().__nonzero__()

    def __or__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.LOGICAL_OR, self, rhs_array)

    def __pos__(self):
        # We know these types are already positive
        if (
            self.dtype.type == np.uint16
            or self.dtype.type == np.uint32
            or self.dtype.type == np.uint64
            or self.dtype.type == np.bool_
        ):
            return self
        return self.perform_unary_op(UnaryOpCode.POSITIVE, self)

    def __pow__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.POWER, self, rhs_array)

    def __radd__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.ADD, lhs_array, self)

    def __rand__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return self.perform_binary_op(
            BinaryOpCode.LOGICAL_AND, lhs_array, self
        )

    def __rdiv__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return lhs_array.internal_truediv(self, inplace=False)

    def __rdivmod__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.DIVMOD, lhs_array, self)

    def __reduce__(self, *args, **kwargs):
        return self.__array__().__reduce__(*args, **kwargs)

    def __reduce_ex__(self, *args, **kwargs):
        return self.__array__().__reduce_ex__(*args, **kwargs)

    def __repr__(self):
        return repr(self.__array__())

    def __rfloordiv__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return self.perform_binary_op(
            BinaryOpCode.FLOOR_DIVIDE, lhs_array, self
        )

    def __rmod__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.MOD, lhs_array, self)

    def __rmul__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.MULTIPLY, lhs_array, self)

    def __ror__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.LOGICAL_OR, lhs_array, self)

    def __rpow__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.POWER, lhs_array, self)

    def __rshift__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.SHIFT_RIGHT, self, rhs_array
        )

    def __rsub__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.SUBTRACT, lhs_array, self)

    def __rtruediv__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return lhs_array.internal_truediv(self, inplace=False)

    def __rxor__(self, lhs):
        lhs_array = self.convert_to_cunumeric_ndarray(lhs)
        return self.perform_binary_op(
            BinaryOpCode.LOGICAL_XOR, lhs_array, self
        )

    # __setattr__

    @add_boilerplate("value", mutates_self=True)
    def __setitem__(self, key, value):
        if key is None:
            raise KeyError("invalid key passed to cunumeric.ndarray")
        if value.dtype != self.dtype:
            temp = ndarray(value.shape, dtype=self.dtype, inputs=(value,))
            temp._thunk.convert(value._thunk)
            value = temp
        key = self._convert_key(key)
        self._thunk.set_item(key, value._thunk)

    def __setstate__(self, state):
        self.__array__().__setstate__(state)

    def __sizeof__(self, *args, **kwargs):
        return self.__array__().__sizeof__(*args, **kwargs)

    def __sub__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.SUBTRACT, self, rhs_array)

    def __str__(self):
        return str(self.__array__())

    def __truediv__(self, rhs):
        return self.internal_truediv(rhs, inplace=False)

    def __xor__(self, rhs):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.LOGICAL_XOR, rhs_array, self
        )

    @add_boilerplate()
    def all(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        return self.perform_unary_reduction(
            UnaryRedCode.ALL,
            self,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            dtype=np.dtype(np.bool_),
            check_types=False,
            initial=initial,
            where=where,
        )

    @add_boilerplate()
    def any(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        return self.perform_unary_reduction(
            UnaryRedCode.ANY,
            self,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            dtype=np.dtype(np.bool_),
            check_types=False,
            initial=initial,
            where=where,
        )

    def argmax(self, axis=None, out=None):
        if self.size == 1:
            return 0
        if axis is None:
            axis = self.ndim - 1
        elif type(axis) != int:
            raise TypeError("'axis' argument for argmax must be an 'int'")
        elif axis < 0 or axis >= self.ndim:
            raise TypeError("invalid 'axis' argument for argmax " + str(axis))
        return self.perform_unary_reduction(
            UnaryRedCode.ARGMAX,
            self,
            axis=axis,
            dtype=np.dtype(np.int64),
            dst=out,
            check_types=False,
        )

    def argmin(self, axis=None, out=None):
        if self.size == 1:
            return 0
        if axis is None:
            axis = self.ndim - 1
        elif type(axis) != int:
            raise TypeError("'axis' argument for argmin must be an 'int'")
        elif axis < 0 or axis >= self.ndim:
            raise TypeError("invalid 'axis' argument for argmin " + str(axis))
        return self.perform_unary_reduction(
            UnaryRedCode.ARGMIN,
            self,
            axis=axis,
            dtype=np.dtype(np.int64),
            dst=out,
            check_types=False,
        )

    def astype(
        self, dtype, order="C", casting="unsafe", subok=True, copy=True
    ):
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self

        casting_allowed = np.can_cast(self.dtype, dtype, casting)
        if casting_allowed:
            # For numeric to non-numeric casting, the dest dtype should be
            # retrived from 'promote_types' to preserve values
            # e.g. ) float 64 to str, np.dtype(dtype) == '<U'
            # this dtype is not safe to store
            if dtype == np.dtype("str"):
                dtype = np.promote_types(self.dtype, dtype)
        else:
            raise TypeError(
                f"Cannot cast array data"
                f"from '{self.dtype}' to '{dtype}' "
                f"to the rule '{casting}'"
            )
        result = ndarray(self.shape, dtype=dtype, inputs=(self,))
        result._thunk.convert(self._thunk, warn=False)
        return result

    def choose(self, choices, out=None, mode="raise"):
        a = self
        if out is not None:
            out = out.convert_to_cunumeric_ndarray(out)

        if isinstance(choices, list):
            choices = tuple(choices)
        is_tuple = isinstance(choices, tuple)
        if is_tuple:
            n = len(choices)
            dtypes = [ch.dtype for ch in choices]
            ch_dtype = np.find_common_type(dtypes, [])
            choices = tuple(
                self.convert_to_cunumeric_ndarray(choices[i]).astype(ch_dtype)
                for i in range(n)
            )

        else:
            choices = self.convert_to_cunumeric_ndarray(choices)
            n = choices.shape[0]
            ch_dtype = choices.dtype
            choices = tuple(choices[i, ...] for i in range(n))

        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError("a array should be integer type")
        if self.dtype is not np.int64:
            a = a.astype(np.int64)
        if mode == "raise":
            if (a < 0).any() | (a >= n).any():
                raise ValueError("invalid entry in choice array")
        elif mode == "wrap":
            a = a % n
        elif mode == "clip":
            a = a.clip(0, n - 1)
        else:
            raise ValueError(
                f"mode={mode} not understood. Must "
                "be 'raise', 'wrap', or 'clip'"
            )

        # we need to broadcast all arrays in choices with
        # input and output arrays
        if out is not None:
            out_shape = broadcast_shapes(a.shape, choices[0].shape, out.shape)
        else:
            out_shape = broadcast_shapes(a.shape, choices[0].shape)

        for c in choices:
            out_shape = broadcast_shapes(out_shape, c.shape)

        # if output is provided, it shape should be the same as out_shape
        if out is not None and out.shape != out_shape:
            raise ValueError(
                f"non-broadcastable output operand with shape "
                f" {str(out.shape)}"
                f" doesn't match the broadcast shape {out_shape}"
            )

        if out is not None and out.dtype == ch_dtype:
            out_arr = out

        else:
            # no output, create one
            out_arr = ndarray(
                shape=out_shape,
                dtype=ch_dtype,
                inputs=(a, choices),
            )

        ch = tuple(c._thunk for c in choices)  #
        out_arr._thunk.choose(
            *ch,
            rhs=a._thunk,
        )
        if out is not None and out.dtype != ch_dtype:
            out._thunk.convert(out_arr._thunk)
            return out
        else:
            return out_arr

    def cholesky(self, no_tril=False):
        input = self
        if input.dtype.kind not in ("f", "c"):
            input = input.astype("float64")
        output = ndarray(
            shape=input.shape,
            dtype=input.dtype,
            inputs=(input,),
        )
        output._thunk.cholesky(input._thunk, no_tril=no_tril)
        return output

    def clip(self, min=None, max=None, out=None):
        args = (
            np.array(min, dtype=self.dtype),
            np.array(max, dtype=self.dtype),
        )
        if args[0].size != 1 or args[1].size != 1:
            runtime.warn(
                "cuNumeric has not implemented clip with array-like "
                "arguments and is falling back to canonical numpy. You "
                "may notice significantly decreased performance for this "
                "function call.",
                category=RuntimeWarning,
            )
            if out is not None:
                self.__array__().clip(min, max, out=out)
                return self.convert_to_cunumeric_ndarray(out, share=True)
            else:
                return self.convert_to_cunumeric_ndarray(
                    self.__array__.clip(min, max)
                )
        return self.perform_unary_op(
            UnaryOpCode.CLIP, self, dst=out, extra_args=args
        )

    def conj(self):
        if self.dtype.kind == "c":
            result = self._thunk.conj()
            return ndarray(self.shape, dtype=self.dtype, thunk=result)
        else:
            return self

    def conjugate(self):
        return self.conj()

    def convolve(self, v, mode):
        assert mode == "same"
        if self.ndim != v.ndim:
            raise RuntimeError("Arrays should have the same dimensions")
        elif self.ndim > 3:
            raise NotImplementedError(
                f"{self.ndim}-D arrays are not yet supported"
            )

        if self.dtype != v.dtype:
            v = v.astype(self.dtype)
        out = ndarray(
            shape=self.shape,
            dtype=self.dtype,
            inputs=(self, v),
        )
        self._thunk.convolve(v._thunk, out._thunk, mode)
        return out

    def copy(self, order="C"):
        # We don't care about dimension order in cuNumeric
        return self.__copy__()

    # diagonal helper. Will return diagonal for arbitrary number of axes;
    # currently offset option is implemented only for the case of number of
    # axes=2. This restriction can be lifted in the future if there is a
    # use case of having arbitrary number of offsets
    def diag_helper(self, offset=0, axes=None, extract=True):
        # diag_helper can be used only for arrays with dim>=1
        if self.ndim < 1:
            raise ValueError("diag_helper is implemented for dim>=1")
        elif self.ndim == 1:
            if axes is not None:
                raise ValueError(
                    "Axes shouldn't be specified when getting "
                    "diagonal for 1D array"
                )
            m = self.shape[0] + np.abs(offset)
            out = ndarray((m, m), dtype=self.dtype, inputs=(self,))
            diag_size = self.shape[0]
            out._thunk.diag_helper(
                self._thunk,
                offset=offset,
                naxes=0,
                extract=False,
            )
        else:
            N = len(axes)
            if len(axes) != len(set(axes)):
                raise ValueError(
                    "axes passed to diag_helper should be all different"
                )
            if self.ndim < N:
                raise ValueError(
                    "Dimension of input array shouldn't be less "
                    "than number of axes"
                )
            # pack the axes that are not going to change
            transpose_axes = tuple(
                ax for ax in range(self.ndim) if ax not in axes
            )
            # only 2 axes provided, we transpose based on the offset
            if N == 2:
                if offset >= 0:
                    a = self.transpose(transpose_axes + (axes[0], axes[1]))
                else:
                    a = self.transpose(transpose_axes + (axes[1], axes[0]))
                    offset = -offset

                if offset >= a.shape[self.ndim - 1]:
                    raise ValueError(
                        "'offset' for diag or diagonal must be in range"
                    )

                diag_size = max(0, min(a.shape[-2], a.shape[-1] - offset))
            # more than 2 axes provided:
            elif N > 2:
                # offsets are supported only when naxes=2
                if offset != 0:
                    raise ValueError(
                        "offset supported for number of axes == 2"
                    )
                # sort axes along which diagonal is calculated by size
                axes = sorted(axes, reverse=True, key=lambda i: self.shape[i])
                axes = tuple(axes)
                # transpose a so axes for which diagonal is calculated are at
                #  at the end
                a = self.transpose(transpose_axes + axes)
                diag_size = a.shape[a.ndim - 1]
            elif N < 2:
                raise ValueError(
                    "number of axes passed to the diag_helper"
                    " should be more than 1"
                )

            tr_shape = tuple(a.shape[i] for i in range(a.ndim - N))
            # calculate shape of the output array
            out_shape = tr_shape + (diag_size,)
            out = ndarray(shape=out_shape, dtype=self.dtype, inputs=(self))

            out._thunk.diag_helper(
                a._thunk,
                offset=offset,
                naxes=N,
                extract=extract,
            )
        return out

    def diagonal(
        self, offset=0, axis1=None, axis2=None, extract=True, axes=None
    ):
        if self.ndim == 1:
            if extract is True:
                raise ValueError("extract can be true only for Ndim >=2")
            axes = None
        else:
            if type(axis1) == int and type(axis2) == int:
                if axes is not None:
                    raise ValueError(
                        "Either axis1/axis2 or axes must be supplied"
                    )
                axes = (axis1, axis2)
            # default values for axes
            elif (axis1 is None) and (axis2 is None) and (axes is None):
                axes = (0, 1)
            elif (axes is not None) and (
                (axis1 is not None) or (axis2 is not None)
            ):
                raise ValueError("Either axis1/axis2 or axes must be supplied")
        return self.diag_helper(offset=offset, axes=axes, extract=extract)

    def dot(self, rhs, out=None):
        rhs_array = self.convert_to_cunumeric_ndarray(rhs)
        if self.size == 1 or rhs_array.size == 1:
            return self.perform_binary_op(
                BinaryOpCode.MULTIPLY,
                self,
                rhs_array,
            )
        out_dtype = self.find_common_type(self, rhs_array)
        # Check for type conversion on the way in
        self_array = self
        if self_array.dtype != out_dtype:
            self_array = ndarray(
                shape=self.shape,
                dtype=out_dtype,
                inputs=(self,),
            )
            self_array._thunk.convert(self._thunk)
        if rhs_array.dtype != out_dtype:
            temp_array = ndarray(
                shape=rhs_array.shape,
                dtype=out_dtype,
                inputs=(rhs_array,),
            )
            temp_array._thunk.convert(rhs_array._thunk)
            rhs_array = temp_array
        # Create output array
        if out is not None:
            out = self.convert_to_cunumeric_ndarray(out, share=True)
            if self.ndim == 1 and rhs_array.ndim == 1:
                if self.shape[0] != rhs.shape[0]:
                    raise ValueError("Dimension mismatch for dot")
                if out.ndim != 0:
                    raise ValueError("Dimension mismatch for dot")
            elif self.ndim == 2 and rhs_array.ndim == 2:
                # Matrix multiply
                if self.shape[1] != rhs_array.shape[0]:
                    raise ValueError("Dimension mismatch for dot")
                if out.shape != (self.shape[0], rhs_array.shape[1]):
                    raise ValueError("Dimension mismatch for dot")
            elif rhs_array.ndim == 1:
                if self.shape[-1] != rhs_array.shape[0]:
                    raise ValueError("Dimension mismatch for dot")
                if out.shape != self.shape[:-1]:
                    raise ValueError("Dimension mismatch for dot")
            else:
                if self.shape[-1] != rhs_array.shape[-2]:
                    raise ValueError("Dimension mismatch for dot")
                if out.shape != (
                    self.shape[:-1]
                    + rhs_array.shape[:-2]
                    + (rhs_array.shape[-1],)
                ):
                    raise ValueError("Dimension mismatch for dot")
        else:
            if self.ndim == 1 and rhs_array.ndim == 1:
                # Inner product
                out = ndarray(
                    shape=(),
                    dtype=out_dtype,
                    inputs=(self_array, rhs_array),
                )
            elif self.ndim == 2 and rhs_array.ndim == 2:
                # Matrix multiply
                if self.shape[1] != rhs_array.shape[0]:
                    raise ValueError("Dimension mismatch for dot")
                out = ndarray(
                    shape=(self.shape[0], rhs_array.shape[1]),
                    dtype=out_dtype,
                    inputs=(self_array, rhs_array),
                )
            elif rhs_array.ndim == 1:
                if self.shape[-1] != rhs_array.shape[0]:
                    raise ValueError("Dimension mismatch for dot")
                out = ndarray(
                    shape=self.shape[:-1],
                    dtype=out_dtype,
                    inputs=(self_array, rhs_array),
                )
            else:
                if self.shape[-1] != rhs_array.shape[-2]:
                    raise ValueError("Dimension mismatch for dot")
                out = ndarray(
                    shape=(
                        self.shape[:-1]
                        + rhs_array.shape[:-2]
                        + (rhs_array.shape[-1],)
                    ),
                    dtype=out_dtype,
                    inputs=(self_array, rhs_array),
                )
        # Perform operation
        out._thunk.dot(self_array._thunk, rhs_array._thunk)
        # Check for type conversion on the way out
        if out.dtype != out_dtype:
            result = ndarray(
                shape=out.shape,
                dtype=out_dtype,
                inputs=(out,),
            )
            result._thunk.convert(out._thunk)
            return result
        else:
            return out

    def dump(self, file):
        self.__array__().dump(file=file)

    def dumps(self):
        return self.__array__().dumps()

    def fill(self, value):
        val = np.array(value, dtype=self.dtype)
        self._thunk.fill(val)

    def getfield(self, dtype, offset=0):
        raise NotImplementedError(
            "cuNumeric does not currently support type reinterpretation "
            "for ndarray.getfield"
        )

    def _convert_singleton_key(self, args: Tuple):
        if len(args) == 0 and self.size == 1:
            return (0,) * self.ndim
        if len(args) == 1 and isinstance(args[0], int):
            flat_idx = args[0]
            result = ()
            for dim_size in reversed(self.shape):
                result = (flat_idx % dim_size,) + result
                flat_idx //= dim_size
            return result
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        if len(args) != self.ndim or any(not isinstance(x, int) for x in args):
            raise KeyError("invalid key")
        return args

    def item(self, *args):
        key = self._convert_singleton_key(args)
        result = self[key]
        assert result.shape == ()
        return result._thunk.__numpy_array__()

    def itemset(self, *args):
        if len(args) == 0:
            raise KeyError("itemset() requires at least one argument")
        value = args[-1]
        args = args[:-1]
        key = self._convert_singleton_key(args)
        self[key] = value

    @add_boilerplate()
    def max(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        return self.perform_unary_reduction(
            UnaryRedCode.MAX,
            self,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    @add_boilerplate()
    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        if axis is not None and type(axis) != int:
            raise TypeError(
                "cunumeric.mean only supports int types for "
                "'axis' currently"
            )
        # Pick our dtype if it wasn't picked yet
        if dtype is None:
            if self.dtype.kind != "f" and self.dtype.kind != "c":
                dtype = np.dtype(np.float64)
            else:
                dtype = self.dtype
        # Do the sum
        if out is not None and out.dtype == dtype:
            sum_array = self.sum(
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=keepdims,
            )
        else:
            sum_array = self.sum(
                axis=axis,
                dtype=dtype,
                keepdims=keepdims,
            )
        if axis is None:
            divisor = reduce(lambda x, y: x * y, self.shape, 1)
        else:
            divisor = self.shape[axis]
        # Divide by the number of things in the collapsed dimensions
        # Pick the right kinds of division based on the dtype
        if dtype.kind == "f":
            sum_array.internal_truediv(
                np.array(divisor, dtype=sum_array.dtype),
                inplace=True,
            )
        else:
            sum_array.__ifloordiv__(np.array(divisor, dtype=sum_array.dtype))
        # Convert to the output we didn't already put it there
        if out is not None and sum_array is not out:
            assert out.dtype != sum_array.dtype
            out._thunk.convert(sum_array._thunk)
            return out
        else:
            return sum_array

    @add_boilerplate()
    def min(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        return self.perform_unary_reduction(
            UnaryRedCode.MIN,
            self,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    @add_boilerplate()
    def prod(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        if self.dtype.type == np.bool_:
            temp = ndarray(
                shape=self.shape,
                dtype=np.dtype(np.int32),
                inputs=(self,),
            )
            temp._thunk.convert(self._thunk)
            self_array = temp
        else:
            self_array = self
        return self.perform_unary_reduction(
            UnaryRedCode.PROD,
            self_array,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def ravel(self, order="C"):
        return self.reshape(-1, order=order)

    def reshape(self, shape, order="C"):
        if shape != -1:
            # Check that these sizes are compatible
            if isinstance(shape, Iterable):
                newsize = 1
                newshape = list()
                unknown_axis = -1
                for ax, dim in enumerate(shape):
                    if dim < 0:
                        newshape.append(np.newaxis)
                        if unknown_axis == -1:
                            unknown_axis = ax
                        else:
                            unknown_axis = -2
                    else:
                        newsize *= dim
                        newshape.append(dim)
                if unknown_axis == -2:
                    raise ValueError("can only specify one unknown dimension")
                if unknown_axis >= 0:
                    if self.size % newsize != 0:
                        raise ValueError(
                            "cannot reshape array of size "
                            + str(self.size)
                            + " into shape "
                            + str(tuple(newshape))
                        )
                    newshape[unknown_axis] = self.size // newsize
                    newsize *= newshape[unknown_axis]
                if newsize != self.size:
                    raise ValueError(
                        "cannot reshape array of size "
                        + str(self.size)
                        + " into shape "
                        + str(shape)
                    )
                shape = tuple(newshape)
            elif isinstance(shape, int):
                if shape != self.size:
                    raise ValueError(
                        "cannot reshape array of size "
                        + str(self.size)
                        + " into shape "
                        + str((shape,))
                    )
            else:
                TypeError("shape must be int-like or tuple-like")
        else:
            # Compute a flattened version of the shape
            shape = (self.size,)
        # Handle an easy case
        if shape == self.shape:
            return self
        return ndarray(
            shape=None,
            thunk=self._thunk.reshape(shape, order),
        )

    def setfield(self, val, dtype, offset=0):
        raise NotImplementedError(
            "cuNumeric does not currently support type reinterpretation "
            "for ndarray.setfield"
        )

    def setflags(self, write=None, align=None, uic=None):
        self.__array__(stacklevel=2).setflags(
            write=write, align=align, uic=uic
        )

    def sort(self, axis=-1, kind="stable", order=None):
        if kind != "stable":
            runtime.warn(
                "cuNumeric uses a different (stable) algorithm than "
                + str(kind)
                + " for sorting",
                category=RuntimeWarning,
                stacklevel=2,
            )
        if order is not None:
            raise NotImplementedError(
                "cuNumeric does not support sorting with 'order' as "
                "ndarray only supports numeric values"
            )
        if axis >= self.ndim or axis < -self.ndim:
            raise ValueError("invalid axis")

        if self._thunk.scalar:
            # nothing to do
            return
        else:
            # this is the default -- sorting of N-D array
            self._thunk.sort(axis=axis)
            return

    def squeeze(self, axis=None):
        if axis is not None:
            if isinstance(axis, int):
                if axis >= self.ndim:
                    raise ValueError(
                        "all axis to squeeze must be less than ndim"
                    )
                if self.shape[axis] != 1:
                    raise ValueError(
                        "axis to squeeze must have extent " "of one"
                    )
            elif isinstance(axis, tuple):
                for ax in axis:
                    if ax >= self.ndim:
                        raise ValueError(
                            "all axes to squeeze must be less than ndim"
                        )
                    if self.shape[ax] != 1:
                        raise ValueError(
                            "all axes to squeeze must have extent of one"
                        )
        return ndarray(shape=None, thunk=self._thunk.squeeze(axis))

    @add_boilerplate()
    def sum(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        if self.dtype.type == np.bool_:
            temp = ndarray(
                shape=self.shape,
                dtype=np.dtype(np.int32),
                inputs=(self,),
            )
            temp._thunk.convert(self._thunk)
            self_array = temp
        else:
            self_array = self
        return self.perform_unary_reduction(
            UnaryRedCode.SUM,
            self_array,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def swapaxes(self, axis1, axis2):
        if axis1 >= self.ndim:
            raise ValueError(
                "axis1=" + str(axis1) + " is too large for swapaxes"
            )
        if axis2 >= self.ndim:
            raise ValueError(
                "axis2=" + str(axis2) + " is too large for swapaxes"
            )
        return ndarray(shape=None, thunk=self._thunk.swapaxes(axis1, axis2))

    def tofile(self, fid, sep="", format="%s"):
        return self.__array__().tofile(fid=fid, sep=sep, format=format)

    def tobytes(self, order="C"):
        return self.__array__().tobytes(order=order)

    def tolist(self):
        return self.__array__().tolist()

    def tostring(self, order="C"):
        return self.__array__().tostring(order=order)

    def transpose(self, axes=None):
        if self.ndim == 1:
            return self
        if axes is None:
            result = ndarray(
                self.shape[::-1],
                dtype=self.dtype,
                inputs=(self,),
            )
            axes = tuple(range(self.ndim - 1, -1, -1))
        elif len(axes) == self.ndim:
            result = ndarray(
                shape=tuple(self.shape[idx] for idx in axes),
                dtype=self.dtype,
                inputs=(self,),
            )
        else:
            raise ValueError(
                "axes must be the same size as ndim for transpose"
            )
        result._thunk.transpose(self._thunk, axes)
        return result

    def flip(self, axis=None):
        result = ndarray(
            shape=self.shape,
            dtype=self.dtype,
            inputs=(self,),
        )
        result._thunk.flip(self._thunk, axis)
        return result

    def view(self, dtype=None, type=None):
        if dtype is not None and dtype != self.dtype:
            raise NotImplementedError(
                "cuNumeric does not currently support type reinterpretation"
            )
        return ndarray(shape=self.shape, dtype=self.dtype, thunk=self._thunk)

    @classmethod
    def get_where_thunk(cls, where, out_shape):
        if where is True:
            return True
        if where is False:
            raise RuntimeError("should have caught this earlier")
        if not isinstance(where, ndarray) or where.dtype != np.bool_:
            raise RuntimeError("should have converted this earlier")
        if where.shape != out_shape:
            raise ValueError("where parameter must have same shape as output")
        return where._thunk

    @staticmethod
    def find_common_type(*args):
        array_types = list()
        scalar_types = list()
        for array in args:
            if array.size == 1:
                scalar_types.append(array.dtype)
            else:
                array_types.append(array.dtype)
        return np.find_common_type(array_types, scalar_types)

    def _maybe_convert(self, dtype, *hints):
        if self.dtype == dtype:
            return self
        copy = ndarray(shape=self.shape, dtype=dtype, inputs=hints)
        copy._thunk.convert(self._thunk)
        return copy

    # For performing normal/broadcast unary operations
    @classmethod
    def perform_unary_op(
        cls,
        op,
        src,
        dst=None,
        extra_args=None,
        dtype=None,
        where=True,
        out_dtype=None,
    ):
        if dst is not None:
            # If the shapes don't match see if we can broadcast
            # This will raise an exception if they can't be broadcast together
            if isinstance(where, ndarray):
                broadcast_shapes(src.shape, dst.shape, where.shape)
            else:
                broadcast_shapes(src.shape, dst.shape)
        else:
            # No output yet, so make one
            if isinstance(where, ndarray):
                out_shape = broadcast_shapes(src.shape, where.shape)
            else:
                out_shape = src.shape
            if dtype is not None:
                dst = ndarray(
                    shape=out_shape,
                    dtype=dtype,
                    inputs=(src, where),
                )
            elif out_dtype is not None:
                dst = ndarray(
                    shape=out_shape,
                    dtype=out_dtype,
                    inputs=(src, where),
                )
            else:
                dst = ndarray(
                    shape=out_shape,
                    dtype=src.dtype
                    if src.dtype.kind != "c"
                    else np.dtype(np.float32)
                    if src.dtype == np.dtype(np.complex64)
                    else np.dtype(np.float64),
                    inputs=(src, where),
                )

        # Quick exit
        if where is False:
            return dst

        op_dtype = (
            dst.dtype
            if out_dtype is None
            and not (op == UnaryOpCode.ABSOLUTE and src.dtype.kind == "c")
            else src.dtype
        )

        if out_dtype is None:
            if dst.dtype != src.dtype and not (
                op == UnaryOpCode.ABSOLUTE and src.dtype.kind == "c"
            ):
                temp = ndarray(
                    dst.shape,
                    dtype=src.dtype,
                    inputs=(src, where),
                )
                temp._thunk.unary_op(
                    op,
                    op_dtype,
                    src._thunk,
                    cls.get_where_thunk(where, dst.shape),
                    extra_args,
                )
                dst._thunk.convert(temp._thunk)
            else:
                dst._thunk.unary_op(
                    op,
                    op_dtype,
                    src._thunk,
                    cls.get_where_thunk(where, dst.shape),
                    extra_args,
                )
        else:
            if dst.dtype != out_dtype:
                temp = ndarray(
                    dst.shape,
                    dtype=out_dtype,
                    inputs=(src, where),
                )
                temp._thunk.unary_op(
                    op,
                    op_dtype,
                    src._thunk,
                    cls.get_where_thunk(where, dst.shape),
                    extra_args,
                )
                dst._thunk.convert(temp._thunk)
            else:
                dst._thunk.unary_op(
                    op,
                    op_dtype,
                    src._thunk,
                    cls.get_where_thunk(where, dst.shape),
                    extra_args,
                )
        return dst

    # For performing reduction unary operations
    @classmethod
    def perform_unary_reduction(
        cls,
        op,
        src,
        axis=None,
        dtype=None,
        dst=None,
        keepdims=False,
        args=None,
        check_types=True,
        initial=None,
        where=True,
    ):
        # TODO: Need to require initial to be given when the array is empty
        #       or a where mask is given.
        if isinstance(where, ndarray):
            # The where array has to broadcast to the src.shape
            if broadcast_shapes(src.shape, where.shape) != src.shape:
                raise ValueError(
                    '"where" array must broadcast against source array '
                    "for reduction"
                )
        # Compute the output shape
        if axis is not None:
            to_reduce = set()
            if type(axis) == int:
                if axis < 0:
                    axis = len(src.shape) + axis
                    if axis < 0:
                        raise ValueError("Illegal 'axis' value")
                elif axis >= src.ndim:
                    raise ValueError("Illegal 'axis' value")
                to_reduce.add(axis)
                axes = (axis,)
            elif type(axis) == tuple:
                for ax in axis:
                    if ax < 0:
                        ax = len(src.shape) + ax
                        if ax < 0:
                            raise ValueError("Illegal 'axis' value")
                    elif ax >= src.ndim:
                        raise ValueError("Illegal 'axis' value")
                    to_reduce.add(ax)
                axes = axis
            else:
                raise TypeError(
                    "Illegal type passed for 'axis' argument "
                    + str(type(axis))
                )
            out_shape = ()
            for dim in range(len(src.shape)):
                if dim in to_reduce:
                    if keepdims:
                        out_shape += (1,)
                else:
                    out_shape += (src.shape[dim],)
        else:
            # Collapsing down to a single value in this case
            out_shape = ()
            axes = None
        # if src.size == 0:
        # return nd
        if dst is None:
            if dtype is not None:
                dst = ndarray(
                    shape=out_shape,
                    dtype=dtype,
                    inputs=(src, where),
                )
            else:
                dst = ndarray(
                    shape=out_shape,
                    dtype=src.dtype,
                    inputs=(src, where),
                )
        else:
            if dtype is not None and dtype != dst.dtype:
                raise TypeError(
                    "Output array type does not match requested dtype"
                )
            if dst.shape != out_shape:
                raise TypeError(
                    "Output array shape "
                    + str(dst.shape)
                    + " does not match expected shape "
                    + str(out_shape)
                )
        # Quick exit
        if where is False:
            return dst
        if check_types and src.dtype != dst.dtype:
            out_dtype = cls.find_common_type(src, dst)
            if src.dtype != out_dtype:
                temp = ndarray(
                    src.shape,
                    dtype=out_dtype,
                    inputs=(src, where),
                )
                temp._thunk.convert(src._thunk)
                src = temp
            if dst.dtype != out_dtype:
                temp = ndarray(
                    dst.shape,
                    dtype=out_dtype,
                    inputs=(src, where),
                )

                temp._thunk.unary_reduction(
                    op,
                    src._thunk,
                    cls.get_where_thunk(where, dst.shape),
                    axes,
                    keepdims,
                    args,
                    initial,
                )
                dst._thunk.convert(temp._thunk)
            else:
                dst._thunk.unary_reduction(
                    op,
                    src._thunk,
                    cls.get_where_thunk(where, dst.shape),
                    axes,
                    keepdims,
                    args,
                    initial,
                )
        else:
            dst._thunk.unary_reduction(
                op,
                src._thunk,
                cls.get_where_thunk(where, dst.shape),
                axes,
                keepdims,
                args,
                initial,
            )
        return dst

    # Return a new cuNumeric array for a binary operation
    @classmethod
    def perform_binary_op(
        cls,
        op,
        one,
        two,
        out=None,
        dtype=None,
        out_dtype=None,
        where=True,
        extra_args=None,
    ):
        args = (one, two, where)

        # Compute the output shape
        shapes = [one.shape, two.shape]
        if isinstance(where, ndarray):
            shapes.append(where.shape)
        if out is not None:
            shapes.append(out.shape)
        out_shape = broadcast_shapes(*shapes)

        if out_dtype is None:
            out_dtype = (
                dtype if dtype is not None else cls.find_common_type(one, two)
            )

        if out is not None:
            if out.shape != out_shape:
                raise ValueError(
                    "non-broadcastable output operand with shape "
                    + str(out.shape)
                    + " doesn't match the broadcast shape "
                    + str(out_shape)
                )
        else:
            out = ndarray(shape=out_shape, dtype=out_dtype, inputs=args)

        # Quick exit
        if where is False:
            return out

        common_type = cls.find_common_type(one, two)
        one = one._maybe_convert(common_type, args)
        two = two._maybe_convert(common_type, args)

        if out.dtype != out_dtype:
            temp = ndarray(shape=out_shape, dtype=out_dtype, inputs=args)
            temp._thunk.binary_op(
                op,
                one._thunk,
                two._thunk,
                cls.get_where_thunk(where, out_shape),
                extra_args,
            )
            out._thunk.convert(temp._thunk)
        else:
            out._thunk.binary_op(
                op,
                one._thunk,
                two._thunk,
                cls.get_where_thunk(where, out_shape),
                extra_args,
            )
        return out

    @classmethod
    def perform_binary_reduction(
        cls,
        op,
        one,
        two,
        dtype,
        extra_args=None,
    ):
        args = (one, two)

        # We only handle bool types here for now
        assert dtype is not None and dtype == np.dtype(np.bool_)
        # Collapsing down to a single value in this case
        # Check to see if we need to broadcast between inputs
        if one.shape != two.shape:
            broadcast = broadcast_shapes(one.shape, two.shape)
        else:
            broadcast = None

        common_type = cls.find_common_type(one, two)
        one = one._maybe_convert(common_type, args)._thunk
        two = two._maybe_convert(common_type, args)._thunk

        dst = ndarray(shape=(), dtype=dtype, inputs=args)
        dst._thunk.binary_reduction(
            op,
            one,
            two,
            broadcast,
            extra_args,
        )
        return dst

    @classmethod
    def perform_where(cls, mask, one, two):
        args = (mask, one, two)

        mask = mask._maybe_convert(np.dtype(np.bool_), args)._thunk

        common_type = cls.find_common_type(one, two)
        one = one._maybe_convert(common_type, args)._thunk
        two = two._maybe_convert(common_type, args)._thunk

        # Compute the output shape
        out_shape = broadcast_shapes(mask.shape, one.shape, two.shape)
        out = ndarray(shape=out_shape, dtype=common_type, inputs=args)
        out._thunk.where(mask, one, two)
        return out
