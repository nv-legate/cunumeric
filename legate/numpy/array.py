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

import functools
import warnings

import numpy as np
import pyarrow
from numpy import broadcast_shapes

from legate.core import LegateArray

from .config import BinaryOpCode, NumPyOpCode, UnaryRedCode
from .doc_utils import copy_docstring
from .runtime import runtime
from .utils import unimplemented

try:
    reduce  # Python 2
except NameError:
    reduce = functools.reduce

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


@copy_docstring(np.ndarray)
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
        stacklevel=2,
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
                    np_array, stacklevel=(stacklevel + 1), share=False
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
        self._thunk.wrap(self)
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
            array = LegateArray(arrow_type, [None, self._thunk])
            self._legate_data = dict()
            self._legate_data["version"] = 1
            data = dict()
            field = pyarrow.field(
                "Legate NumPy Array", arrow_type, nullable=False
            )
            data[field] = array
            self._legate_data["data"] = data
        return self._legate_data

    # A class method for sanitizing inputs by converting them to
    # Legate ndarray types
    @staticmethod
    def convert_to_legate_ndarray(obj, stacklevel=2, share=False):
        # If this is an instance of one of our ndarrays then we're done
        if isinstance(obj, ndarray):
            return obj
        # Ask the runtime to make a numpy thunk for this object
        thunk = runtime.get_numpy_thunk(
            obj, stacklevel=(stacklevel + 1), share=share
        )
        return ndarray(shape=None, stacklevel=(stacklevel + 1), thunk=thunk)

    @staticmethod
    def convert_to_predicate_ndarray(obj, stacklevel):
        # Keep all boolean types as they are
        if obj is True or obj is False:
            return obj
        if isinstance(obj, ndarray):
            thunk = obj._thunk
        else:
            thunk = runtime.get_numpy_thunk(obj, stacklevel=(stacklevel + 1))
        if thunk.size == 1:
            # Convert this into a bool for now, in the future we may want to
            # defer this anyway to avoid blocking deferred execution
            return bool(thunk.get_scalar_array(stacklevel=(stacklevel + 1)))
        else:
            # If the type of the thunk is not bool then we need to convert it
            if obj.dtype != np.bool_:
                result = ndarray(
                    thunk.shape,
                    dtype=np.dtype(np.bool_),
                    stacklevel=(stacklevel + 1),
                )
                result._thunk.convert(
                    thunk, warn=True, stacklevel=(stacklevel + 1)
                )
                return result
            else:
                return ndarray(
                    shape=None, stacklevel=(stacklevel + 1), thunk=thunk
                )

    # Properties for ndarray

    # Disable these since they seem to cause problems
    # when our arrays do not last long enough, instead
    # users will go through the __array__ method

    # @property
    # def __array_interface__(self):
    #    return self.__array__(stacklevel=2).__array_interface__

    # @property
    # def __array_priority__(self):
    #    return self.__array__(stacklevel=2).__array_priority__

    # @property
    # def __array_struct__(self):
    #    return self.__array__(stacklevel=2).__array_struct__

    @property
    def T(self):
        return self.transpose(stacklevel=2)

    @property
    def base(self):
        return self.__array__(stacklevel=2).base

    @property
    def data(self):
        return self.__array__(stacklevel=2).data

    @property
    def dtype(self):
        return self._thunk.dtype

    @property
    def flags(self):
        return self.__array__(stacklevel=2).flags

    @property
    def flat(self):
        return self.__array__(stacklevel=2).flat

    @property
    def imag(self):
        return ndarray(shape=None, thunk=self._thunk.imag(stacklevel=2))

    @property
    def ndim(self):
        return self._thunk.ndim

    @property
    def real(self):
        return ndarray(shape=None, thunk=self._thunk.real(stacklevel=2))

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
        return self.__array__(stacklevel=2).strides

    @property
    def ctypes(self):
        return self.__array__(stacklevel=2).ctypes

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
        return self.perform_unary_op(NumPyOpCode.ABSOLUTE, self)

    def __add__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.ADD, self, rhs_array)

    def __and__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(NumPyOpCode.LOGICAL_AND, rhs_array)

    def __array__(self, dtype=None, stacklevel=1):
        if dtype is None:
            return self._thunk.__numpy_array__(stacklevel=(stacklevel + 1))
        else:
            return self._thunk.__numpy_array__(
                stacklevel=(stacklevel + 1)
            ).__array__(dtype)

    # def __array_prepare__(self, *args, **kwargs):
    #    return self.__array__(stacklevel=2).__array_prepare__(*args, **kwargs)

    # def __array_wrap__(self, *args, **kwargs):
    #    return self.__array__(stacklevel=2).__array_wrap__(*args, **kwargs)

    def __bool__(self):
        if self.size == 1:
            return bool(self._thunk.get_scalar_array(stacklevel=2))
        else:
            return bool(self.__array__(stacklevel=2))

    def __complex__(self):
        if self.size == 1:
            return complex(self._thunk.get_scalar_array(stacklevel=2))
        else:
            return complex(self.__array__(stacklevel=2))

    def __contains__(self, item):
        if isinstance(item, np.ndarray):
            args = (item,)
        else:  # Otherwise convert it to a scalar numpy array of our type
            args = (np.array(item, dtype=self.dtype),)
        if args[0].size != 1:
            raise ValueError("contains needs scalar item")
        return self.perform_unary_reduction(
            NumPyOpCode.CONTAINS,
            NumPyOpCode.SUM_RADIX,
            self,
            axis=None,
            dtype=np.dtype(np.bool_),
            args=args,
            check_types=False,
            stacklevel=2,
        )

    def __copy__(self):
        result = ndarray(self.shape, self.dtype)
        result._thunk.copy(self._thunk, deep=False, stacklevel=2)
        return result

    def __deepcopy__(self, memo=None):
        result = ndarray(self.shape, self.dtype)
        result._thunk.copy(self._thunk, deep=True, stacklevel=2)
        return result

    def __div__(self, rhs):
        return self.internal_truediv(rhs, inplace=False, stacklevel=2)

    def __divmod__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(NumPyOpCode.DIVMOD, self, rhs_array)

    def __eq__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.EQUAL, self, rhs_array, out_dtype=np.dtype(np.bool_)
        )

    def __float__(self):
        if self.size == 1:
            return float(self._thunk.get_scalar_array(stacklevel=2))
        else:
            return float(self.__array__(stacklevel=2))

    def __floordiv__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.FLOOR_DIVIDE, self, rhs_array
        )

    def __format__(self, *args, **kwargs):
        return self.__array__(stacklevel=2).__format__(*args, **kwargs)

    def __ge__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.GREATER_EQUAL,
            self,
            rhs_array,
            out_dtype=np.dtype(np.bool_),
        )

    # __getattribute__

    def _convert_key(self, key, stacklevel=2, first=True):
        # Convert any arrays stored in a key to a legate array
        if key is None or key is Ellipsis:
            return key
        if np.isscalar(key):
            return key
        if isinstance(key, slice):
            return key
        if isinstance(key, tuple) and first:
            result = ()
            for k in key:
                result += (
                    self._convert_key(
                        k, stacklevel=(stacklevel + 1), first=False
                    ),
                )
            return result
        # Otherwise convert it to a legate array and get the thunk
        return self.convert_to_legate_ndarray(
            key, stacklevel=(stacklevel + 1)
        )._thunk

    def __getitem__(self, key):
        if key is None:
            raise KeyError("invalid key passed to legate.numpy.ndarray")
        # If we're a scalar, we're our own value
        if self.size == 1:
            if (self.ndim == 0 and key != () and key != Ellipsis) or (
                key != ((0,) * self.ndim)
            ):
                raise KeyError("invalid key passed to legate.numpy.ndarray")
            return self
        key = self._convert_key(key)
        return ndarray(
            shape=None, thunk=self._thunk.get_item(key, stacklevel=2)
        )

    def __gt__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.GREATER, self, rhs_array, out_dtype=np.dtype(np.bool_)
        )

    def __hash__(self, *args, **kwargs):
        raise TypeError("unhashable type: legate.numpy.ndarray")

    def __iadd__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(BinaryOpCode.ADD, self, rhs_array, out=self)
        return self

    def __iand__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(
            NumPyOpCode.LOGICAL_AND, self, rhs_array, out=self
        )
        return self

    def __idiv__(self, rhs):
        return self.internal_truediv(rhs, inplace=True, stacklevel=2)

    def __idivmod__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(NumPyOpCode.DIVMOD, self, rhs_array, out=self)
        return self

    def __ifloordiv__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(
            BinaryOpCode.FLOOR_DIVIDE, self, rhs_array, out=self
        )
        return self

    def __ilshift__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(
            NumPyOpCode.SHIFT_LEFT, self, rhs_array, out=self
        )
        return self

    def __imod__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(NumPyOpCode.MODULUS, self, rhs_array, out=self)
        return self

    def __imul__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(
            BinaryOpCode.MULTIPLY, self, rhs_array, out=self
        )
        return self

    def __int__(self):
        if self.size == 1:
            return int(self._thunk.get_scalar_array(stacklevel=2))
        else:
            return int(self.__array__(stacklevel=2))

    def __invert__(self):
        if self.dtype == np.bool_:
            # Boolean values are special, just do logical NOT
            return self.perform_unary_op(
                NumPyOpCode.LOGICAL_NOT, self, out_dtype=np.dtype(np.bool_)
            )
        else:
            return self.perform_unary_op(NumPyOpCode.INVERT, self)

    def __ior__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(
            NumPyOpCode.LOGICAL_OR, self, rhs_array, out=self
        )
        return self

    def __ipow__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(BinaryOpCode.POWER, self, rhs_array, out=self)
        return self

    def __irshift__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(
            NumPyOpCode.SHIFT_RIGHT, self, rhs_array, out=self
        )
        return self

    def __iter__(self):
        return self.__array__(stacklevel=2).__iter__()

    def __isub__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(
            BinaryOpCode.SUBTRACT, self, rhs_array, out=self
        )
        return self

    def internal_truediv(self, rhs, inplace, stacklevel):
        rhs_array = self.convert_to_legate_ndarray(
            rhs, stacklevel=(stacklevel + 1)
        )
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
                stacklevel=(stacklevel + 1),
            )
            temp._thunk.convert(
                self_array._thunk, warn=False, stacklevel=(stacklevel + 1)
            )
            self_array = temp
        if rhs_array.dtype != common_type:
            temp = ndarray(
                rhs_array.shape, dtype=common_type, stacklevel=(stacklevel + 1)
            )
            temp._thunk.convert(
                rhs_array._thunk, warn=False, stacklevel=(stacklevel + 1)
            )
            rhs_array = temp
        return self.perform_binary_op(
            BinaryOpCode.DIVIDE,
            self_array,
            rhs_array,
            out=self if inplace else None,
            stacklevel=(stacklevel + 1),
        )

    def __itruediv__(self, rhs):
        return self.internal_truediv(rhs, inplace=True, stacklevel=2)

    def __ixor__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        self.perform_binary_op(
            NumPyOpCode.LOGICAL_XOR, self, rhs_array, out=self
        )
        return self

    def __le__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.LESS_EQUAL,
            self,
            rhs_array,
            out_dtype=np.dtype(np.bool_),
        )

    def __len__(self):
        return self.shape[0]

    def __lshift__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(NumPyOpCode.SHIFT_LEFT, self, rhs_array)

    def __lt__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(
            BinaryOpCode.LESS, self, rhs_array, out_dtype=np.dtype(np.bool_)
        )

    def __matmul__(self, value):
        return self.dot(value, stacklevel=2)

    def __mod__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.MOD, self, rhs_array)

    def __mul__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.MULTIPLY, self, rhs_array)

    def __ne__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
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
        return self.perform_unary_op(NumPyOpCode.NEGATIVE, self)

    # __new__

    def count_nonzero(self, stacklevel, axis):
        # unary_reduction will reduce size 1 arrays to themeslves for
        # deferred arrays.  Because of that, we need to handle size 1
        # explicitly.
        if self.size == 1:
            return self.convert_to_legate_ndarray(
                int(
                    self._thunk.get_scalar_array(
                        stacklevel=(stacklevel + 1)
                    ).item()
                    != 0
                )
            )
        if self.size == 0:
            return 0
        return ndarray.perform_unary_reduction(
            NumPyOpCode.COUNT_NONZERO,
            NumPyOpCode.SUM_RADIX,
            self,
            axis=axis,
            dtype=np.dtype(np.uint64),
            stacklevel=(stacklevel + 1),
            check_types=False,
        )

    def nonzero(self, stacklevel=1):
        # This case is handled at the high level.  Deferred nonzero expects
        # the input to not have size 1 (future base).
        if self.size == 1:
            ndim = self.ndim + int(self.ndim == 0)
            if self._thunk.get_scalar_array(stacklevel=stacklevel + 1) == 0:
                return (
                    ndarray(
                        shape=(0,),
                        thunk=runtime.find_or_create_array_thunk(
                            np.array([]),
                            stacklevel=stacklevel + 1,
                            share=False,
                        ),
                    ),
                ) * ndim
            else:
                return (
                    ndarray(
                        shape=(1,),
                        thunk=runtime.find_or_create_array_thunk(
                            np.array([0]),
                            stacklevel=stacklevel + 1,
                            share=False,
                        ),
                    ),
                ) * ndim
        thunks = self._thunk.nonzero(stacklevel=stacklevel + 1)
        result = ()
        for thunk in thunks:
            result += (
                ndarray(shape=thunk.shape, thunk=thunk, dtype=thunk.dtype),
            )
        return result

    def __nonzero__(self):
        if self.size == 1:
            return bool(self._thunk.get_scalar_array(stacklevel=2) != 0)
        else:
            return self.__array__(stacklevel=2).__nonzero__()

    def __pos__(self):
        # We know these types are already positive
        if (
            self.dtype.type == np.uint16
            or self.dtype.type == np.uint32
            or self.dtype.type == np.uint64
            or self.dtype.type == np.bool_
        ):
            return self
        return self.perform_unary_op(NumPyOpCode.POSITIVE, self)

    def __pow__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.POWER, self, rhs_array)

    def __radd__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.ADD, lhs_array, self)

    def __rand__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return self.perform_binary_op(NumPyOpCode.LOGICAL_AND, lhs_array, self)

    def __rdiv__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return lhs_array.internal_truediv(self, inplace=False, stacklevel=2)

    def __rdivmod__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return self.perform_binary_op(NumPyOpCode.DIVMOD, lhs_array, self)

    def __reduce__(self, *args, **kwargs):
        return self.__array__(stacklevel=2).__reduce__(*args, **kwargs)

    def __reduce_ex__(self, *args, **kwargs):
        return self.__array__(stacklevel=2).__reduce_ex__(*args, **kwargs)

    def __repr__(self):
        if self.size == 1:
            return repr(self._thunk.get_scalar_array(stacklevel=2))
        else:
            return repr(self.__array__(stacklevel=2))

    def __rfloordiv__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return self.perform_binary_op(
            BinaryOpCode.FLOOR_DIVIDE, lhs_array, self
        )

    def __rmod__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.MOD, lhs_array, self)

    def __rmul__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.MULTIPLY, lhs_array, self)

    def __ror__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return self.perform_binary_op(NumPyOpCode.LOGICAL_OR, lhs_array, self)

    def __rpow__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.POWER, lhs_array, self)

    def __rshift__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(NumPyOpCode.SHIFT_RIGHT, self, rhs_array)

    def __rsub__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return self.perform_binary_op(BinaryOpCode.SUBTRACT, lhs_array, self)

    def __rtruediv__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return lhs_array.internal_truediv(self, inplace=False, stacklevel=2)

    def __rxor__(self, lhs):
        lhs_array = self.convert_to_legate_ndarray(lhs)
        return self.perform_binary_op(NumPyOpCode.LOGICAL_XOR, lhs_array, self)

    # __setattr__

    def __setitem__(self, key, value):
        if key is None:
            raise KeyError("invalid key passed to legate.numpy.ndarray")
        value_array = self.convert_to_legate_ndarray(value)
        if value_array.dtype != self.dtype:
            temp = ndarray(value_array.shape, dtype=self.dtype)
            temp._thunk.convert(value_array._thunk, stacklevel=2)
            value_array = temp
        if self.size == 1:
            if (self.ndim == 0 and key != () and key != Ellipsis) or (
                key != ((0,) * self.ndim)
            ):
                raise KeyError("invalid key passed to legate.numpy.ndarray")
        key = self._convert_key(key)
        self._thunk.set_item(key, value_array._thunk, stacklevel=2)

    def __setstate__(self, state):
        self.__array__(stacklevel=2).__setstate__(state)

    def __sizeof__(self, *args, **kwargs):
        return self.__array__(stacklevel=2).__sizeof__(*args, **kwargs)

    def __sub__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(BinaryOpCode.SUBTRACT, self, rhs_array)

    def __str__(self):
        if self.size == 1:
            return str(
                np.reshape(
                    self._thunk.get_scalar_array(stacklevel=2), self.shape
                )
            )
        else:
            return str(self.__array__(stacklevel=2))

    def __truediv__(self, rhs, stacklevel=1):
        return self.internal_truediv(
            rhs, inplace=False, stacklevel=(stacklevel + 1)
        )

    def __xor__(self, rhs):
        rhs_array = self.convert_to_legate_ndarray(rhs)
        return self.perform_binary_op(NumPyOpCode.LOGICAL_XOR, rhs_array, self)

    @unimplemented
    def all(self, axis=None, out=None, keepdims=False):
        numpy_array = self.__array__(stacklevel=3).all(
            axis=axis, out=out, keepdims=keepdims
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    @unimplemented
    def any(self, axis=None, out=None, keepdims=False):
        numpy_array = self.__array__(stacklevel=3).any(
            axis=axis, out=out, keepdims=keepdims
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def argmax(self, axis=None, out=None, stacklevel=1):
        if self.size == 1:
            return 0
        if axis is None:
            axis = self.ndim - 1
        elif type(axis) != int:
            raise TypeError("'axis' argument for argmax must be an 'int'")
        elif axis < 0 or axis >= self.ndim:
            raise TypeError("invalid 'axis' argument for argmax " + str(axis))
        if out is not None:
            out = self.convert_to_legate_ndarray(
                out, stacklevel=(stacklevel + 1), share=True
            )
        return self.perform_unary_reduction(
            NumPyOpCode.ARGMAX,
            NumPyOpCode.ARGMAX_RADIX,
            self,
            axis=axis,
            dtype=np.dtype(np.int64),
            dst=out,
            check_types=False,
            stacklevel=(stacklevel + 1),
        )

    def argmin(self, axis=None, out=None, stacklevel=1):
        if self.size == 1:
            return 0
        if axis is None:
            axis = self.ndim - 1
        elif type(axis) != int:
            raise TypeError("'axis' argument for argmin must be an 'int'")
        elif axis < 0 or axis >= self.ndim:
            raise TypeError("invalid 'axis' argument for argmin " + str(axis))
        if out is not None:
            out = self.convert_to_legate_ndarray(
                out, stacklevel=(stacklevel + 1), share=True
            )
        return self.perform_unary_reduction(
            NumPyOpCode.ARGMIN,
            NumPyOpCode.ARGMIN_RADIX,
            self,
            axis=axis,
            dtype=np.dtype(np.int64),
            dst=out,
            check_types=False,
            stacklevel=(stacklevel + 1),
        )

    @unimplemented
    def argpartition(self, kth, axis=-1, kind="introselect", order=None):
        numpy_array = self.__array__(stacklevel=3).argpartition(
            kth=kth, axis=axis, kind=kind, order=order
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    @unimplemented
    def argsort(self, axis=-1, kind=None, order=None):
        numpy_array = self.__array__(stacklevel=3).argsort(
            axis=axis, kind=kind, order=order
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def astype(
        self, dtype, order="C", casting="unsafe", subok=True, copy=True
    ):
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self
        result = ndarray(self.shape, dtype=dtype)
        result._thunk.convert(self._thunk, warn=False, stacklevel=2)
        return result

    @unimplemented
    def byteswap(self, inplace=False):
        if inplace:
            self.__array__(stacklevel=3).byteswap(inplace=True)
            return self
        else:
            numpy_array = self.__array__(stacklevel=3).byteswap(inplace=False)
            return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    @unimplemented
    def choose(self, choices, out, mode="raise"):
        numpy_array = self.__array__(stacklevel=3).choose(
            choices=choices, out=out, mode=mode
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def clip(self, min=None, max=None, out=None):
        args = (
            np.array(min, dtype=self.dtype),
            np.array(max, dtype=self.dtype),
        )
        if (args[0] is not None and args[0].size != 1) or (
            args[1] is not None and args[1].size != 1
        ):
            warnings.warn(
                "legate.numpy has not implemented clip with array-like "
                "arguments and is falling back to canonical numpy. You "
                "may notice significantly decreased performance for this "
                "function call.",
                stacklevel=2,
                category=RuntimeWarning,
            )
            if out is not None:
                self.__array__(stacklevel=2).clip(min, max, out=out)
                return self.convert_to_legate_ndarray(
                    out, stacklevel=2, share=True
                )
            else:
                return self.convert_to_legate_ndarray(
                    self.__array__.clip(min, max)
                )
        return self.perform_unary_op(
            NumPyOpCode.CLIP, self, dst=out, args=args
        )

    @unimplemented
    def compress(self, condition, axis=None, out=None):
        numpy_array = self.__array__(stacklevel=3).compress(
            condition, axis=axis, out=out
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    @unimplemented
    def conj(self, stacklevel=1):
        numpy_array = self.__array__(stacklevel=3).conj()
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    @unimplemented
    def conjugate(self):
        numpy_array = self.__array__(stacklevel=3).conj()
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def copy(self, order="C"):
        # We don't care about dimension order in legate
        return self.__copy__()

    @unimplemented
    def cumprod(self, axis=None, dtype=None, out=None):
        numpy_array = self.__array__(stacklevel=3).cumprod(
            axis=axis, dtype=dtype, out=out
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    @unimplemented
    def cumsum(self, axis=None, dtype=None, out=None):
        numpy_array = self.__array__(stacklevel=3).cumsum(
            axis=axis, dtype=dtype, out=out
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    @unimplemented
    def diagonal(self, offset=0, axis1=0, axis2=1):
        numpy_array = self.__array__(stacklevel=3).diagonal(
            offset=offset, axis1=axis1, axis2=axis2
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def dot(self, rhs, out=None, stacklevel=1):
        rhs_array = self.convert_to_legate_ndarray(
            rhs, stacklevel=(stacklevel + 1)
        )
        if self.size == 1 or rhs_array.size == 1:
            return self.perform_binary_op(
                BinaryOpCode.MULTIPLY,
                self,
                rhs_array,
                stacklevel=(stacklevel + 1),
            )
        out_dtype = self.find_common_type(self, rhs_array)
        if out is not None:
            out = self.convert_to_legate_ndarray(
                out, stacklevel=(stacklevel + 1), share=True
            )
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
                    shape=(), dtype=out_dtype, stacklevel=(stacklevel + 1)
                )
            elif self.ndim == 2 and rhs_array.ndim == 2:
                # Matrix multiply
                if self.shape[1] != rhs_array.shape[0]:
                    raise ValueError("Dimension mismatch for dot")
                out = ndarray(
                    shape=(self.shape[0], rhs_array.shape[1]),
                    dtype=out_dtype,
                    stacklevel=(stacklevel + 1),
                )
            elif rhs_array.ndim == 1:
                if self.shape[-1] != rhs_array.shape[0]:
                    raise ValueError("Dimension mismatch for dot")
                out = ndarray(
                    shape=self.shape[:-1],
                    dtype=out_dtype,
                    stacklevel=(stacklevel + 1),
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
                    stacklevel=(stacklevel + 1),
                )
        # Check for type conversion on the way in
        self_array = self
        if self_array.dtype != out_dtype:
            self_array = ndarray(
                shape=self.shape, dtype=out_dtype, stacklevel=(stacklevel + 1)
            )
            self_array._thunk.convert(self._thunk, stacklevel=(stacklevel + 1))
        if rhs_array.dtype != out_dtype:
            temp_array = ndarray(
                shape=rhs_array.shape,
                dtype=out_dtype,
                stacklevel=(stacklevel + 1),
            )
            temp_array._thunk.convert(
                rhs_array._thunk, stacklevel=(stacklevel + 1)
            )
            rhs_array = temp_array
        out._thunk.dot(
            self_array._thunk, rhs_array._thunk, stacklevel=(stacklevel + 1)
        )
        # Check type conversion on the way out
        if out.dtype != out_dtype:
            result = ndarray(
                shape=out.shape, dtype=out_dtype, stacklevel=(stacklevel + 1)
            )
            result._thunk.convert(out._thunk, stacklevel=(stacklevel + 1))
            return result
        else:
            return out

    def dump(self, file):
        self.__array__(stacklevel=2).dump(file=file)

    def dumps(self):
        return self.__array__(stacklevel=2).dumps()

    def fill(self, value, stacklevel=1):
        val = np.array(value, dtype=self.dtype)
        self._thunk.fill(val, stacklevel=stacklevel + 1)

    @unimplemented
    def flatten(self, order="C"):
        numpy_array = self.__array__(stacklevel=3).flatten(order=order)
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def getfield(self, dtype, offset=0):
        raise NotImplementedError(
            "legate.numpy does not currently support type reinterpretation "
            "for ndarray.getfield"
        )

    def item(self, *args):
        if args is None:
            raise KeyError("invalid key passed to legate.numpy.ndarray")
        key = self._convert_key(args)
        result = ndarray(
            shape=None, thunk=self._thunk.get_item(key, stacklevel=2)
        )
        assert result.size == 1
        if self.dtype.kind == "f":
            return float(result)
        elif self.dtype.kind == "i" or self.dtype.kind == "u":
            return int(result)
        elif self.dtype.kind == "b":
            return bool(result)
        else:
            raise TypeError("Invalid dtype")

    def itemset(self, key, value=0):
        if key is None:
            raise KeyError("invalid key passed to legate.numpy.ndarray")
        key = self._convert_key(key)
        val = np.array(value, dtype=self.dtype)
        value_array = self.convert_to_legate_ndarray(val)
        self._thunk.set_item(key, value_array._thunk, stacklevel=2)

    def max(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
        stacklevel=1,
    ):
        where = self.convert_to_predicate_ndarray(
            where, stacklevel=(stacklevel + 1)
        )
        if self.size == 1:
            if out is not None:
                self.copy(out)
                return out
            else:
                return self
        if out is not None:
            out = self.convert_to_legate_ndarray(
                out, stacklevel=(stacklevel + 1), share=True
            )
        return self.perform_unary_reduction(
            UnaryRedCode.MAX,
            NumPyOpCode.MAX_RADIX,
            self,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            stacklevel=(stacklevel + 1),
        )

    def mean(
        self, axis=None, dtype=None, out=None, keepdims=False, stacklevel=1
    ):
        if axis is not None and type(axis) != int:
            raise TypeError(
                "legate.numpy.mean only supports int types for "
                "'axis' currently"
            )
        # Pick our dtype if it wasn't picked yet
        if dtype is None:
            if self.dtype.kind != "f" and self.dtype.kind != "c":
                dtype = np.dtype(np.float64)
            else:
                dtype = self.dtype
        if out is not None:
            out = ndarray.convert_to_legate_ndarray(
                out, stacklevel=(stacklevel + 1), share=True
            )
        # Do the sum
        if out is not None and out.dtype == dtype:
            sum_array = self.sum(
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=keepdims,
                stacklevel=(stacklevel + 1),
            )
        else:
            sum_array = self.sum(
                axis=axis,
                dtype=dtype,
                keepdims=keepdims,
                stacklevel=(stacklevel + 1),
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
                stacklevel=(stacklevel + 1),
            )
        else:
            sum_array.__ifloordiv__(np.array(divisor, dtype=sum_array.dtype))
        # Convert to the output we didn't already put it there
        if out is not None and sum_array is not out:
            assert out.dtype != sum_array.dtype
            out._thunk.convert(sum_array._thunk, stacklevel=(stacklevel + 1))
            return out
        else:
            return sum_array

    def min(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
        stacklevel=1,
    ):
        where = self.convert_to_predicate_ndarray(
            where, stacklevel=(stacklevel + 1)
        )
        if self.size == 1:
            if out is not None:
                self.copy(out)
                return out
            else:
                return self
        if out is not None:
            out = self.convert_to_legate_ndarray(
                out, stacklevel=(stacklevel + 1), share=True
            )
        return self.perform_unary_reduction(
            UnaryRedCode.MIN,
            NumPyOpCode.MIN_RADIX,
            self,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            stacklevel=(stacklevel + 1),
        )

    @unimplemented
    def partition(self, kth, axis=-1, kind="introselect", order=None):
        self.__array__(stacklevel=3).partition(
            kth=kth, axis=axis, kind=kind, order=order
        )

    def prod(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
        stacklevel=1,
    ):
        if self.size == 1:
            if out is not None:
                self.copy(out)
                return out
            else:
                return self
        self_array = self
        where = self.convert_to_predicate_ndarray(
            where, stacklevel=(stacklevel + 1)
        )
        if self.dtype.type == np.bool_:
            temp = ndarray(
                shape=self.shape,
                dtype=np.dtype(np.int32),
                stacklevel=(stacklevel + 1),
            )
            temp._thunk.convert(self._thunk, stacklevel=(stacklevel + 1))
            self_array = temp
        if out is not None:
            out = self.convert_to_legate_ndarray(
                out, stacklevel=(stacklevel + 1), share=True
            )
        return self.perform_unary_reduction(
            UnaryRedCode.PROD,
            NumPyOpCode.PROD_RADIX,
            self_array,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            stacklevel=(stacklevel + 1),
        )

    @unimplemented
    def ptp(self, axis=None, out=None, keepdims=False):
        numpy_array = self.__array__(stacklevel=3).ptp(
            axis=axis, out=out, keepdims=keepdims
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    @unimplemented
    def put(self, indices, values, mode="raise"):
        self.__array__(stacklevel=3).put(
            indices=indices, values=values, mode=mode
        )

    def ravel(self, order="C", stacklevel=1):
        return self.reshape(-1, order=order, stacklevel=(stacklevel + 1))

    @unimplemented
    def repeat(self, repeats, axis=None):
        numpy_array = self.__array__(stacklevel=3).repeat(repeats, axis=axis)
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def reshape(self, shape, order="C", stacklevel=1):
        if shape != -1:
            # Check that these sizes are compatible
            if isinstance(shape, tuple):
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
            thunk=self._thunk.reshape(
                shape, order, stacklevel=(stacklevel + 1)
            ),
        )

    @unimplemented
    def resize(self, new_shape, refcheck=True):
        numpy_array = self.__array__(stacklevel=3).resize(
            new_shape=new_shape, refcheck=refcheck
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    @unimplemented
    def round(self, decimals=0, out=None):
        numpy_array = self.__array__(stacklevel=3).round(
            decimals=decimals, out=out
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    @unimplemented
    def searchsorted(self, v, side="left", sorter=None):
        numpy_array = self.__array__(stacklevel=3).searchsorted(
            v=v, side=side, sorter=sorter
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def setfield(self, val, dtype, offset=0):
        raise NotImplementedError(
            "legate.numpy does not currently support type reinterpretation "
            "for ndarray.setfield"
        )

    def setflags(self, write=None, align=None, uic=None):
        self.__array__(stacklevel=2).setflags(
            write=write, align=align, uic=uic
        )

    @unimplemented
    def sort(self, axis=-1, kind=None, order=None):
        numpy_array = self.__array__(stacklevel=3).sort(
            axis=axis, kind=kind, order=order
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

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
        return ndarray(
            shape=None, thunk=self._thunk.squeeze(axis, stacklevel=2)
        )

    @unimplemented
    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        numpy_array = self.__array__(stacklevel=3).std(
            axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def sum(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
        stacklevel=1,
    ):
        if self.size == 1:
            if out is not None:
                self.copy(out)
                return out
            else:
                return np.copy(
                    self._thunk.get_scalar_array(stacklevel=stacklevel + 1)
                )
        self_array = self
        where = self.convert_to_predicate_ndarray(
            where, stacklevel=(stacklevel + 1)
        )
        if self.dtype.type == np.bool_:
            temp = ndarray(
                shape=self.shape,
                dtype=np.dtype(np.int32),
                stacklevel=(stacklevel + 1),
            )
            temp._thunk.convert(self._thunk, stacklevel=(stacklevel + 1))
            self_array = temp
        if out is not None:
            out = self.convert_to_legate_ndarray(
                out, stacklevel=(stacklevel + 1), share=True
            )
        return self.perform_unary_reduction(
            UnaryRedCode.SUM,
            NumPyOpCode.SUM_RADIX,
            self_array,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            stacklevel=(stacklevel + 1),
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
        return ndarray(
            shape=None, thunk=self._thunk.swapaxes(axis1, axis2, stacklevel=2)
        )

    @unimplemented
    def take(self, indices, axis=None, out=None, mode="raise"):
        numpy_array = self.__array__(stacklevel=3).take(
            indices=indices, axis=axis, out=out, mode=mode
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def tofile(self, fid, sep="", format="%s"):
        return self.__array__(stacklevel=2).tofile(
            fid=fid, sep=sep, format=format
        )

    def tobytes(self, order="C"):
        return self.__array__(stacklevel=2).tobytes(order=order)

    def tolist(self):
        return self.__array__(stacklevel=2).tolist()

    def tostring(self, order="C"):
        return self.__array__(stacklevel=2).tostring(order=order)

    @unimplemented
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        numpy_array = self.__array__(stacklevel=3).trace(
            offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def transpose(self, axes=None, stacklevel=1):
        if axes is None:
            result = ndarray(
                self.shape[::-1], dtype=self.dtype, stacklevel=(stacklevel + 1)
            )
            axes = tuple(range(self.ndim - 1, -1, -1))
        elif len(axes) == self.ndim:
            result = ndarray(
                shape=tuple(map(lambda x, y: x[y], self.shape, axes)),
                dtype=self.dtype,
                stacklevel=(stacklevel + 1),
            )
        else:
            raise ValueError(
                "axes must be the same size as ndim for transpose"
            )
        result._thunk.transpose(self._thunk, axes, stacklevel=(stacklevel + 1))
        return result

    @unimplemented
    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        numpy_array = self.__array__(stacklevel=3).var(
            axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims
        )
        return self.convert_to_legate_ndarray(numpy_array, stacklevel=3)

    def view(self, dtype=None, type=None):
        if dtype is not None and dtype != self.dtype:
            raise NotImplementedError(
                "legate.numpy does not currently support type reinterpretation"
            )
        return ndarray(shape=self.shape, dtype=self.dtype, thunk=self._thunk)

    @classmethod
    def get_where_thunk(cls, where, out_shape, stacklevel):
        if where is True:
            return True
        elif where is False:
            raise RuntimeError("should have caught this earlier")
        else:
            array = cls.convert_to_legate_array(where)
            if array.shape != out_shape:
                raise ValueError(
                    "where parameter must have same shape as output"
                )
            # Convert this to a bool array if needed
            if array.dtype != np.bool_:
                temp = ndarray(
                    shape=array.shape,
                    dtype=np.dtype(np.bool_),
                    stacklevel=(stacklevel + 1),
                )
                temp._thunk.convert(array._thunk, stacklevel=(stacklevel + 1))
                array = temp
            return array._thunk

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

    # For performing normal/broadcast unary operations
    @classmethod
    def perform_unary_op(
        cls,
        op,
        src,
        dst=None,
        args=None,
        dtype=None,
        where=True,
        out_dtype=None,
        check_types=True,
        stacklevel=2,
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
                    stacklevel=(stacklevel + 1),
                    inputs=(src, where),
                )
            elif out_dtype is not None:
                dst = ndarray(
                    shape=out_shape,
                    dtype=out_dtype,
                    stacklevel=(stacklevel + 1),
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
                    stacklevel=(stacklevel + 1),
                    inputs=(src, where),
                )
        # Quick exit
        if where is False:
            return dst
        op_dtype = (
            dst.dtype
            if out_dtype is None
            and not (op == NumPyOpCode.ABSOLUTE and src.dtype.kind == "c")
            else src.dtype
        )
        if check_types:
            if out_dtype is None:
                if dst.dtype != src.dtype and not (
                    op == NumPyOpCode.ABSOLUTE and src.dtype.kind == "c"
                ):
                    temp = ndarray(
                        dst.shape,
                        dtype=src.dtype,
                        stacklevel=(stacklevel + 1),
                        inputs=(src, where),
                    )
                    temp._thunk.unary_op(
                        op,
                        op_dtype,
                        src._thunk,
                        cls.get_where_thunk(
                            where, dst.shape, stacklevel=(stacklevel + 1)
                        ),
                        args,
                        stacklevel=(stacklevel + 1),
                    )
                    dst._thunk.convert(
                        temp._thunk, stacklevel=(stacklevel + 1)
                    )
                else:
                    dst._thunk.unary_op(
                        op,
                        op_dtype,
                        src._thunk,
                        cls.get_where_thunk(
                            where, dst.shape, stacklevel=(stacklevel + 1)
                        ),
                        args,
                        stacklevel=(stacklevel + 1),
                    )
            else:
                if dst.dtype != out_dtype:
                    temp = ndarray(
                        dst.shape,
                        dtype=out_dtype,
                        stacklevel=(stacklevel + 1),
                        inputs=(src, where),
                    )
                    temp._thunk.unary_op(
                        op,
                        op_dtype,
                        src._thunk,
                        cls.get_where_thunk(
                            where, dst.shape, stacklevel=(stacklevel + 1)
                        ),
                        args,
                        stacklevel=(stacklevel + 1),
                    )
                    dst._thunk.convert(
                        temp._thunk, stacklevel=(stacklevel + 1)
                    )
                else:
                    dst._thunk.unary_op(
                        op,
                        op_dtype,
                        src._thunk,
                        cls.get_where_thunk(
                            where, dst.shape, stacklevel=(stacklevel + 1)
                        ),
                        args,
                        stacklevel=(stacklevel + 1),
                    )
        else:
            dst._thunk.unary_op(
                op,
                op_dtype,
                src._thunk,
                cls.get_where_thunk(
                    where, dst.shape, stacklevel=(stacklevel + 1)
                ),
                args,
                stacklevel=(stacklevel + 1),
            )
        return dst

    # For performing reduction unary operations
    @classmethod
    def perform_unary_reduction(
        cls,
        op,
        redop,
        src,
        axis=None,
        dtype=None,
        dst=None,
        keepdims=False,
        args=None,
        check_types=True,
        initial=None,
        where=True,
        stacklevel=2,
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
            for dim in xrange(len(src.shape)):
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
                    stacklevel=(stacklevel + 1),
                    inputs=(src, where),
                )
            else:
                dst = ndarray(
                    shape=out_shape,
                    dtype=src.dtype,
                    stacklevel=(stacklevel + 1),
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
                    stacklevel=(stacklevel + 1),
                    inputs=(src, where),
                )
                temp._thunk.convert(src._thunk, stacklevel=(stacklevel + 1))
                src = temp
            if dst.dtype != out_dtype:
                temp = ndarray(
                    dst.shape,
                    dtype=out_dtype,
                    stacklevel=(stacklevel + 1),
                    inputs=(src, where),
                )

                temp._thunk.unary_reduction(
                    op,
                    redop,
                    src._thunk,
                    cls.get_where_thunk(
                        where, dst.shape, stacklevel=(stacklevel + 1)
                    ),
                    axes,
                    keepdims,
                    args,
                    initial,
                    stacklevel=(stacklevel + 1),
                )
                dst._thunk.convert(temp._thunk, stacklevel=(stacklevel + 1))
            else:
                dst._thunk.unary_reduction(
                    op,
                    redop,
                    src._thunk,
                    cls.get_where_thunk(
                        where, dst.shape, stacklevel=(stacklevel + 1)
                    ),
                    axes,
                    keepdims,
                    args,
                    initial,
                    stacklevel=(stacklevel + 1),
                )
        else:
            dst._thunk.unary_reduction(
                op,
                redop,
                src._thunk,
                cls.get_where_thunk(
                    where, dst.shape, stacklevel=(stacklevel + 1)
                ),
                axes,
                keepdims,
                args,
                initial,
                stacklevel=(stacklevel + 1),
            )
        return dst

    # Return a new legate array for a binary operation
    @classmethod
    def perform_binary_op(
        cls,
        op,
        one,
        two,
        out=None,
        dtype=None,
        args=None,
        where=True,
        out_dtype=None,
        check_types=True,
        stacklevel=2,
    ):
        # Compute the output shape
        if out is None:
            # Compute the output shape and confirm any broadcasting
            if isinstance(where, ndarray):
                out_shape = broadcast_shapes(one.shape, two.shape, where.shape)
            else:
                out_shape = broadcast_shapes(one.shape, two.shape)
            if dtype is not None:
                out = ndarray(
                    shape=out_shape,
                    dtype=dtype,
                    stacklevel=(stacklevel + 1),
                    inputs=(one, two, where),
                )
            elif out_dtype is not None:
                out = ndarray(
                    shape=out_shape,
                    dtype=out_dtype,
                    stacklevel=(stacklevel + 1),
                    inputs=(one, two, where),
                )
            else:
                out_dtype = cls.find_common_type(one, two)
                out = ndarray(
                    shape=out_shape,
                    dtype=out_dtype,
                    stacklevel=(stacklevel + 1),
                    inputs=(one, two, where),
                )
        else:
            if isinstance(where, ndarray):
                out_shape = broadcast_shapes(
                    one.shape, two.shape, out.shape, where.shape
                )
            else:
                out_shape = broadcast_shapes(one.shape, two.shape, out.shape)
            if out.shape != out_shape:
                raise ValueError(
                    "non-broadcastable output operand with shape "
                    + str(out.shape)
                    + " doesn't match the broadcast shape "
                    + str(out_shape)
                )
        # Quick exit
        if where is False:
            return out
        if out_dtype is None:
            out_dtype = cls.find_common_type(one, two)
        if check_types:
            if one.dtype != two.dtype:
                common_type = cls.find_common_type(one, two)
                if one.dtype != common_type:
                    temp = ndarray(
                        shape=one.shape,
                        dtype=common_type,
                        stacklevel=(stacklevel + 1),
                        inputs=(one, two, where),
                    )
                    temp._thunk.convert(
                        one._thunk, stacklevel=(stacklevel + 1)
                    )
                    one = temp
                if two.dtype != common_type:
                    temp = ndarray(
                        shape=two.shape,
                        dtype=common_type,
                        stacklevel=(stacklevel + 1),
                        inputs=(one, two, where),
                    )
                    temp._thunk.convert(
                        two._thunk, stacklevel=(stacklevel + 1)
                    )
                    two = temp
            if out.dtype != out_dtype:
                temp = ndarray(
                    shape=out.shape,
                    dtype=out_dtype,
                    stacklevel=(stacklevel + 1),
                    inputs=(one, two, where),
                )
                temp._thunk.binary_op(
                    op,
                    one._thunk,
                    two._thunk,
                    cls.get_where_thunk(
                        where, out.shape, stacklevel=(stacklevel + 1)
                    ),
                    args,
                    stacklevel=(stacklevel + 1),
                )
                out._thunk.convert(temp._thunk, stacklevel=(stacklevel + 1))
            else:
                out._thunk.binary_op(
                    op,
                    one._thunk,
                    two._thunk,
                    cls.get_where_thunk(
                        where, out.shape, stacklevel=(stacklevel + 1)
                    ),
                    args,
                    stacklevel=(stacklevel + 1),
                )
        else:
            out._thunk.binary_op(
                op,
                one._thunk,
                two._thunk,
                cls.get_where_thunk(
                    where, out.shape, stacklevel=(stacklevel + 1)
                ),
                args,
                stacklevel=(stacklevel + 1),
            )
        return out

    @classmethod
    def perform_binary_reduction(
        cls,
        op,
        one,
        two,
        dtype=None,
        args=None,
        check_types=True,
        stacklevel=2,
    ):
        # We only handle bool types here for now
        assert dtype is not None and dtype == np.dtype(np.bool_)
        # Collapsing down to a single value in this case
        # Check to see if we need to broadcast between inputs
        if one.shape != two.shape:
            broadcast = broadcast_shapes(one.shape, two.shape)
        else:
            broadcast = None
        dst = ndarray(
            shape=(),
            dtype=dtype,
            stacklevel=(stacklevel + 1),
            inputs=(one, two),
        )
        if check_types and one.dtype != two.dtype:
            if one.dtype != two.dtype:
                common_type = cls.find_common_type(one, two)
                if one.dtype != common_type:
                    temp = ndarray(
                        shape=one.shape,
                        dtype=common_type,
                        stacklevel=(stacklevel + 1),
                        inputs=(one, two),
                    )
                    temp._thunk.convert(
                        one._thunk, stacklevel=(stacklevel + 1)
                    )
                    one = temp
                if two.dtype != common_type:
                    temp = ndarray(
                        shape=two.shape,
                        dtype=common_type,
                        stacklevel=(stacklevel + 1),
                        inputs=(one, two),
                    )
                    temp._thunk.convert(
                        two._thunk, stacklevel=(stacklevel + 1)
                    )
                    two = temp
        dst._thunk.binary_reduction(
            op,
            one._thunk,
            two._thunk,
            broadcast,
            args,
            stacklevel=(stacklevel + 1),
        )
        return dst

    @classmethod
    def perform_ternary_op(
        cls,
        op,
        one,
        two,
        three,
        out=None,
        dtype=None,
        args=None,
        where=True,
        out_dtype=None,
        check_types=True,
        stacklevel=2,
    ):
        # Compute the output shape
        if out is None:
            # Compute the output shape and confirm any broadcasting
            if isinstance(where, ndarray):
                out_shape = broadcast_shapes(
                    one.shape, two.shape, three.shape, where.shape
                )
            else:
                out_shape = broadcast_shapes(one.shape, two.shape, three.shape)
            if dtype is not None:
                out = ndarray(
                    shape=out_shape,
                    dtype=dtype,
                    stacklevel=(stacklevel + 1),
                    inputs=(one, two, three, where),
                )
            elif out_dtype is not None:
                out = ndarray(
                    shape=out_shape,
                    dtype=out_dtype,
                    stacklevel=(stacklevel + 1),
                    inputs=(one, two, three, where),
                )
            else:
                out = ndarray(
                    shape=out_shape,
                    dtype=np.result_type(one, two, three),
                    stacklevel=(stacklevel + 1),
                    inputs=(one, two, three, where),
                )
        else:
            if isinstance(where, ndarray):
                out_shape = broadcast_shapes(
                    one.shape, two.shape, three.shape, out.shape, where.shape
                )
            else:
                out_shape = broadcast_shapes(
                    one.shape, two.shape, three.shape, out.shape
                )
            if out.shape != out_shape:
                raise ValueError(
                    "out array shape "
                    + str(out.shape)
                    + " does not match expected shape "
                    + str(out_shape)
                )
        # Quick exit
        if where is False:
            return out
        if out_dtype is None:
            out_dtype = np.result_type(one, two, three)
        if check_types:
            if one.dtype != two.dtype or one.dtype != three.dtype:
                common_type = cls.find_common_type(one, two, three)
                if one.dtype != common_type:
                    temp = ndarray(
                        shape=one.shape,
                        dtype=common_type,
                        stacklevel=(stacklevel + 1),
                        inputs=(one, two, three, where),
                    )
                    temp._thunk.convert(
                        one._thunk, stacklevel=(stacklevel + 1)
                    )
                    one = temp
                if two.dtype != common_type:
                    temp = ndarray(
                        shape=two.shape,
                        dtype=common_type,
                        stacklevel=(stacklevel + 1),
                        inputs=(one, two, three, where),
                    )
                    temp._thunk.convert(
                        two._thunk, stacklevel=(stacklevel + 1)
                    )
                    two = temp
                if three.dtype != common_type:
                    temp = ndarray(
                        shape=three.shape,
                        dtype=common_type,
                        stacklevel=(stacklevel + 1),
                        inputs=(one, two, three, where),
                    )
                    temp._thunk.convert(
                        three._thunk, stacklevel=(stacklevel + 1)
                    )
                    three = temp
            if out.dtype != out_dtype:
                temp = ndarray(
                    shape=out.shape,
                    dtype=out_dtype,
                    stacklevel=(stacklevel + 1),
                    inputs=(one, two, three, where),
                )
                temp._thunk.ternary_op(
                    op,
                    one._thunk,
                    two._thunk,
                    three._thunk,
                    cls.get_where_thunk(
                        where, out.shape, stacklevel=(stacklevel + 1)
                    ),
                    args,
                    stacklevel=(stacklevel + 1),
                )
                out._thunk.convert(temp._thunk, stacklevel=(stacklevel + 1))
            else:
                out._thunk.ternary_op(
                    op,
                    one._thunk,
                    two._thunk,
                    three._thunk,
                    cls.get_where_thunk(
                        where, out.shape, stacklevel=(stacklevel + 1)
                    ),
                    args,
                    stacklevel=(stacklevel + 1),
                )
        else:
            out._thunk.ternary_op(
                op,
                one._thunk,
                two._thunk,
                three._thunk,
                cls.get_where_thunk(
                    where, out.shape, stacklevel=(stacklevel + 1)
                ),
                args,
                stacklevel=(stacklevel + 1),
            )
        return out
