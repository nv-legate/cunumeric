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

from collections.abc import Iterable
from functools import reduce, wraps
from inspect import signature
from typing import Optional, Set, Tuple

import numpy as np
import pyarrow

from legate.core import Array

from .config import FFTDirection, FFTNormalization, UnaryOpCode, UnaryRedCode
from .coverage import clone_class
from .runtime import runtime
from .utils import broadcast_shapes, dot_modes


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

        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            assert (where_idx is None or len(args) <= where_idx) and (
                out_idx is None or len(args) <= out_idx
            ), "'where' and 'out' should be passed as keyword arguments"

            # Convert relevant arguments to cuNumeric ndarrays
            args = tuple(
                convert_to_cunumeric_ndarray(arg)
                if idx in indices and arg is not None
                else arg
                for (idx, arg) in enumerate(args)
            )
            for (k, v) in kwargs.items():
                if v is None:
                    continue
                elif k == "where":
                    kwargs[k] = convert_to_predicate_ndarray(v)
                elif k == "out":
                    kwargs[k] = convert_to_cunumeric_ndarray(v, share=True)
                elif k in keys:
                    kwargs[k] = convert_to_cunumeric_ndarray(v)

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
                result = convert_to_cunumeric_ndarray(res_scalar)
                if out is not None:
                    out._thunk.copy(result._thunk)
                    result = out
                return result

            return func(*args, **kwargs)

        return wrapper

    return decorator


def convert_to_cunumeric_ndarray(obj, share=False):
    # If this is an instance of one of our ndarrays then we're done
    if isinstance(obj, ndarray):
        return obj
    # Ask the runtime to make a numpy thunk for this object
    thunk = runtime.get_numpy_thunk(obj, share=share)
    return ndarray(shape=None, thunk=thunk)


def convert_to_predicate_ndarray(obj):
    # Keep all boolean types as they are
    if obj is True or obj is False:
        return obj
    # GH #135
    raise NotImplementedError(
        "the `where` parameter is currently not supported"
    )


@clone_class(np.ndarray)
class ndarray:
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
        # `inputs` being a cuNumeric ndarray is definitely a bug
        assert not isinstance(inputs, ndarray)
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
        """

        The transposed array.

        Same as ``self.transpose()``.

        See Also
        --------
        cunumeric.transpose
        ndarray.transpose

        """
        return self.transpose()

    @property
    def base(self):
        """
        Returns dtype for the base element of the subarrays,
        regardless of their dimension or shape.

        See Also
        --------
        numpy.dtype.subdtype

        """
        return self.__array__().base

    @property
    def data(self):
        """
        Python buffer object pointing to the start of the array's data.

        """
        return self.__array__().data

    @property
    def dtype(self):
        """
        Data-type of the array's elements.

        See Also
        --------
        astype : Cast the values contained in the array to a new data-type.
        view : Create a view of the same data but a different data-type.
        numpy.dtype

        """
        return self._thunk.dtype

    @property
    def flags(self):
        """
        Information about the memory layout of the array.

        Attributes
        ----------
        C_CONTIGUOUS (C)
            The data is in a single, C-style contiguous segment.
        F_CONTIGUOUS (F)
            The data is in a single, Fortran-style contiguous segment.
        OWNDATA (O)
            The array owns the memory it uses or borrows it from another
            object.
        WRITEABLE (W)
            The data area can be written to.  Setting this to False locks
            the data, making it read-only.  A view (slice, etc.) inherits
            WRITEABLE from its base array at creation time, but a view of a
            writeable array may be subsequently locked while the base array
            remains writeable. (The opposite is not true, in that a view of a
            locked array may not be made writeable.  However, currently,
            locking a base object does not lock any views that already
            reference it, so under that circumstance it is possible to alter
            the contents of a locked array via a previously created writeable
            view onto it.)  Attempting to change a non-writeable array raises
            a RuntimeError exception.
        ALIGNED (A)
            The data and all elements are aligned appropriately for the
            hardware.
        WRITEBACKIFCOPY (X)
            This array is a copy of some other array. The C-API function
            PyArray_ResolveWritebackIfCopy must be called before deallocating
            to the base array will be updated with the contents of this array.
        FNC
            F_CONTIGUOUS and not C_CONTIGUOUS.
        FORC
            F_CONTIGUOUS or C_CONTIGUOUS (one-segment test).
        BEHAVED (B)
            ALIGNED and WRITEABLE.
        CARRAY (CA)
            BEHAVED and C_CONTIGUOUS.
        FARRAY (FA)
            BEHAVED and F_CONTIGUOUS and not C_CONTIGUOUS.

        Notes
        -----
        The `flags` object can be accessed dictionary-like (as in
        ``a.flags['WRITEABLE']``), or by using lowercased attribute names (as
        in ``a.flags.writeable``). Short flag names are only supported in
        dictionary access.

        Only the WRITEBACKIFCOPY, WRITEABLE, and ALIGNED flags can be
        changed by the user, via direct assignment to the attribute or
        dictionary entry, or by calling `ndarray.setflags`.

        The array flags cannot be set arbitrarily:
        - WRITEBACKIFCOPY can only be set ``False``.
        - ALIGNED can only be set ``True`` if the data is truly aligned.
        - WRITEABLE can only be set ``True`` if the array owns its own memory
        or the ultimate owner of the memory exposes a writeable buffer
        interface or is a string.

        Arrays can be both C-style and Fortran-style contiguous
        simultaneously. This is clear for 1-dimensional arrays, but can also
        be true for higher dimensional arrays.

        Even for contiguous arrays a stride for a given dimension
        ``arr.strides[dim]`` may be *arbitrary* if ``arr.shape[dim] == 1``
        or the array has no elements.
        It does not generally hold that ``self.strides[-1] == self.itemsize``
        for C-style contiguous arrays or ``self.strides[0] == self.itemsize``
        for Fortran-style contiguous arrays is true.
        """
        return self.__array__().flags

    @property
    def flat(self):
        """
        A 1-D iterator over the array.

        See Also
        --------
        flatten : Return a copy of the array collapsed into one dimension.

        """
        return self.__array__().flat

    @property
    def imag(self):
        """
        The imaginary part of the array.

        """
        if self.dtype.kind == "c":
            return ndarray(shape=self.shape, thunk=self._thunk.imag())
        else:
            result = ndarray(self.shape, self.dtype)
            result.fill(0)
            return result

    @property
    def ndim(self):
        """
        Number of array dimensions.

        """
        return self._thunk.ndim

    @property
    def real(self):
        """

        The real part of the array.

        """
        if self.dtype.kind == "c":
            return ndarray(shape=self.shape, thunk=self._thunk.real())
        else:
            return self

    @property
    def shape(self):
        """

        Tuple of array dimensions.

        See Also
        --------
        shape : Equivalent getter function.
        reshape : Function forsetting ``shape``.
        ndarray.reshape : Method for setting ``shape``.

        """
        return self._thunk.shape

    @property
    def size(self):
        """

        Number of elements in the array.

        Equal to ``np.prod(a.shape)``, i.e., the product of the array's
        dimensions.

        Notes
        -----
        `a.size` returns a standard arbitrary precision Python integer. This
        may not be the case with other methods of obtaining the same value
        (like the suggested ``np.prod(a.shape)``, which returns an instance
        of ``np.int_``), and may be relevant if the value is used further in
        calculations that may overflow a fixed size integer type.

        """
        s = 1
        if self.ndim == 0:
            return s
        for p in self.shape:
            s *= p
        return s

    @property
    def itemsize(self):
        """

        The element size of this data-type object.

        For 18 of the 21 types this number is fixed by the data-type.
        For the flexible data-types, this number can be anything.

        """
        return self._thunk.dtype.itemsize

    @property
    def nbytes(self):
        """

        Total bytes consumed by the elements of the array.

        Notes
        -----
        Does not include memory consumed by non-element attributes of the
        array object.

        """
        return self.itemsize * self.size

    @property
    def strides(self):
        """

        Tuple of bytes to step in each dimension when traversing an array.

        The byte offset of element ``(i[0], i[1], ..., i[n])`` in an array
        `a` is::

            offset = sum(np.array(i) * a.strides)

        A more detailed explanation of strides can be found in the
        "ndarray.rst" file in the NumPy reference guide.

        Notes
        -----
        Imagine an array of 32-bit integers (each 4 bytes)::

            x = np.array([[0, 1, 2, 3, 4],
                         [5, 6, 7, 8, 9]], dtype=np.int32)

        This array is stored in memory as 40 bytes, one after the other
        (known as a contiguous block of memory).  The strides of an array tell
        us how many bytes we have to skip in memory to move to the next
        position along a certain axis.  For example, we have to skip 4 bytes
        (1 value) to move to the next column, but 20 bytes (5 values) to get
        to the same position in the next row.  As such, the strides for the
        array `x` will be ``(20, 4)``.

        """
        return self.__array__().strides

    @property
    def ctypes(self):
        """

        An object to simplify the interaction of the array with the ctypes
        module.

        This attribute creates an object that makes it easier to use arrays
        when calling shared libraries with the ctypes module. The returned
        object has, among others, data, shape, and strides attributes (see
        :external+numpy:attr:`numpy.ndarray.ctypes` for details) which
        themselves return ctypes objects that can be used as arguments to a
        shared library.

        Parameters
        ----------
        None

        Returns
        -------
        c : object
            Possessing attributes data, shape, strides, etc.

        """
        return self.__array__().ctypes

    # Methods for ndarray

    def __abs__(self):
        """a.__abs__(/)

        Return ``abs(self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # Handle the nice case of it being unsigned
        from ._ufunc import absolute

        return absolute(self)

    def __add__(self, rhs):
        """a.__add__(value, /)

        Return ``self+value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import add

        return add(self, rhs)

    def __and__(self, rhs):
        """a.__and__(value, /)

        Return ``self&value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import logical_and

        return logical_and(self, rhs)

    def __array__(self, dtype=None):
        """a.__array__([dtype], /)

        Returns either a new reference to self if dtype is not given or a new
        array of provided data type if dtype is different from the current
        dtype of the array.

        """
        if dtype is None:
            return self._thunk.__numpy_array__()
        else:
            return self._thunk.__numpy_array__().__array__(dtype)

    # def __array_prepare__(self, *args, **kwargs):
    #    return self.__array__().__array_prepare__(*args, **kwargs)

    # def __array_wrap__(self, *args, **kwargs):
    #    return self.__array__().__array_wrap__(*args, **kwargs)

    def __bool__(self):
        """a.__bool__(/)

        Return ``self!=0``

        """
        return bool(self.__array__())

    def __complex__(self):
        """a.__complex__(/)"""
        return complex(self.__array__())

    def __contains__(self, item):
        """a.__contains__(key, /)

        Return ``key in self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if isinstance(item, np.ndarray):
            args = (item.astype(self.dtype),)
        else:  # Otherwise convert it to a scalar numpy array of our type
            args = (np.array(item, dtype=self.dtype),)
        if args[0].size != 1:
            raise ValueError("contains needs scalar item")
        return self._perform_unary_reduction(
            UnaryRedCode.CONTAINS,
            self,
            axis=None,
            dtype=np.dtype(np.bool_),
            args=args,
            check_types=False,
        )

    def __copy__(self):
        """a.__copy__()

        Used if :func:`copy.copy` is called on an array. Returns a copy
        of the array.

        Equivalent to ``a.copy(order='K')``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        result = ndarray(self.shape, self.dtype, inputs=(self,))
        result._thunk.copy(self._thunk, deep=False)
        return result

    def __deepcopy__(self, memo=None):
        """a.__deepcopy__(memo, /)

        Deep copy of array.

        Used if :func:`copy.deepcopy` is called on an array.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        result = ndarray(self.shape, self.dtype, inputs=(self,))
        result._thunk.copy(self._thunk, deep=True)
        return result

    def __div__(self, rhs):
        """a.__div__(value, /)

        Return ``self/value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__truediv__(rhs)

    def __divmod__(self, rhs):
        """a.__divmod__(value, /)

        Return ``divmod(self, value)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        raise NotImplementedError(
            "cunumeric.ndarray doesn't support __divmod__ yet"
        )

    def __eq__(self, rhs):
        """a.__eq__(value, /)

        Return ``self==value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import equal

        return equal(self, rhs)

    def __float__(self):
        """a.__float__(/)

        Return ``float(self)``.

        """
        return float(self.__array__())

    def __floordiv__(self, rhs):
        """a.__floordiv__(value, /)

        Return ``self//value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import floor_divide

        return floor_divide(self, rhs)

    def __format__(self, *args, **kwargs):
        return self.__array__().__format__(*args, **kwargs)

    def __ge__(self, rhs):
        """a.__ge__(value, /)

        Return ``self>=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import greater_equal

        return greater_equal(self, rhs)

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
            # Otherwise convert it to a cuNumeric array, check types
            # and get the thunk
            key = convert_to_cunumeric_ndarray(key)
            if key.dtype != bool and not np.issubdtype(key.dtype, np.integer):
                raise TypeError("index arrays should be int or bool type")
            if key.dtype != bool and key.dtype != np.int64:
                runtime.warn(
                    "converting index array to int64 type",
                    category=RuntimeWarning,
                )
                key = key.astype(np.int64)

            return key._thunk

    @add_boilerplate()
    def __getitem__(self, key):
        """a.__getitem__(key, /)

        Return ``self[key]``.

        """
        key = self._convert_key(key)
        return ndarray(shape=None, thunk=self._thunk.get_item(key))

    def __gt__(self, rhs):
        """a.__gt__(value, /)

        Return ``self>value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import greater

        return greater(self, rhs)

    def __hash__(self, *args, **kwargs):
        raise TypeError("unhashable type: cunumeric.ndarray")

    def __iadd__(self, rhs):
        """a.__iadd__(value, /)

        Return ``self+=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import add

        return add(self, rhs, out=self)

    def __iand__(self, rhs):
        """a.__iand__(value, /)

        Return ``self&=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import logical_and

        return logical_and(self, rhs, out=self)

    def __idiv__(self, rhs):
        """a.__idiv__(value, /)

        Return ``self/=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__itruediv__(rhs)

    def __ifloordiv__(self, rhs):
        """a.__ifloordiv__(value, /)

        Return ``self//=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import floor_divide

        return floor_divide(self, rhs, out=self)

    def __ilshift__(self, rhs):
        """a.__ilshift__(value, /)

        Return ``self<<=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import left_shift

        return left_shift(self, rhs, out=self)

    def __imod__(self, rhs):
        """a.__imod__(value, /)

        Return ``self%=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import remainder

        return remainder(self, rhs, out=self)

    def __imul__(self, rhs):
        """a.__imul__(value, /)

        Return ``self*=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import multiply

        return multiply(self, rhs, out=self)

    def __int__(self):
        """a.__int__(/)

        Return ``int(self)``.

        """
        return int(self.__array__())

    def __invert__(self):
        """a.__invert__(/)

        Return ``~self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.dtype == np.bool_:
            # Boolean values are special, just do logical NOT
            from ._ufunc import logical_not

            return logical_not(self)
        else:
            from ._ufunc import invert

            return invert(self)

    def __ior__(self, rhs):
        """a.__ior__(/)

        Return ``self|=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import logical_or

        return logical_or(self, rhs, out=self)

    def __ipow__(self, rhs):
        """a.__ipow__(/)

        Return ``self**=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import power

        return power(self, rhs, out=self)

    def __irshift__(self, rhs):
        """a.__irshift__(/)

        Return ``self>>=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import right_shift

        return right_shift(self, rhs, out=self)

    def __iter__(self):
        """a.__iter__(/)"""
        return self.__array__().__iter__()

    def __isub__(self, rhs):
        """a.__isub__(/)

        Return ``self-=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import subtract

        return subtract(self, rhs, out=self)

    def __itruediv__(self, rhs):
        """a.__itruediv__(/)

        Return ``self/=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import true_divide

        return true_divide(self, rhs, out=self)

    def __ixor__(self, rhs):
        """a.__ixor__(/)

        Return ``self^=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import logical_xor

        return logical_xor(self, rhs, out=self)

    def __le__(self, rhs):
        """a.__le__(value, /)

        Return ``self<=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import less_equal

        return less_equal(self, rhs)

    def __len__(self):
        """a.__len__(/)

        Return ``len(self)``.

        """
        return self.shape[0]

    def __lshift__(self, rhs):
        """a.__lshift__(value, /)

        Return ``self<<value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import left_shift

        return left_shift(self, rhs)

    def __lt__(self, rhs):
        """a.__lt__(value, /)

        Return ``self<value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import less

        return less(self, rhs)

    def __matmul__(self, value):
        """a.__matmul__(value, /)

        Return ``self@value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.dot(value)

    def __mod__(self, rhs):
        """a.__mod__(value, /)

        Return ``self%value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import remainder

        return remainder(self, rhs)

    def __mul__(self, rhs):
        """a.__mul__(value, /)

        Return ``self*value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import multiply

        return multiply(self, rhs)

    def __ne__(self, rhs):
        """a.__ne__(value, /)

        Return ``self!=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import not_equal

        return not_equal(self, rhs)

    def __neg__(self):
        """a.__neg__(value, /)

        Return ``-self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import negative

        return negative(self)

    # __new__

    @add_boilerplate()
    def nonzero(self):
        """a.nonzero()

        Return the indices of the elements that are non-zero.

        Refer to :func:`cunumeric.nonzero` for full documentation.

        See Also
        --------
        cunumeric.nonzero : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        thunks = self._thunk.nonzero()
        return tuple(
            ndarray(shape=thunk.shape, thunk=thunk) for thunk in thunks
        )

    def __nonzero__(self):
        """a.nonzero(/)

        Return the indices of the elements that are non-zero.

        Refer to :func:`cunumeric.nonzero` for full documentation.

        See Also
        --------
        cunumeric.nonzero : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__array__().__nonzero__()

    def __or__(self, rhs):
        """a.__or__(value, /)

        Return ``self|value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import logical_or

        return logical_or(self, rhs)

    def __pos__(self):
        """a.__pos__(value, /)

        Return ``+self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # the positive opeartor is equivalent to copy
        from ._ufunc import positive

        return positive(self)

    def __pow__(self, rhs):
        """a.__pow__(value, /)

        Return ``pow(self, value)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import power

        return power(self, rhs)

    def __radd__(self, lhs):
        """a.__radd__(value, /)

        Return ``value+self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import add

        return add(lhs, self)

    def __rand__(self, lhs):
        """a.__rand__(value, /)

        Return ``value&self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import logical_and

        return logical_and(lhs, self)

    def __rdiv__(self, lhs):
        """a.__rdiv__(value, /)

        Return ``value/self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import true_divide

        return true_divide(lhs, self)

    def __rdivmod__(self, lhs):
        """a.__rdivmod__(value, /)

        Return ``divmod(value, self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        raise NotImplementedError(
            "cunumeric.ndarray doesn't support __rdivmod__ yet"
        )

    def __reduce__(self, *args, **kwargs):
        """a.__reduce__(/)

        For pickling.

        """
        return self.__array__().__reduce__(*args, **kwargs)

    def __reduce_ex__(self, *args, **kwargs):
        return self.__array__().__reduce_ex__(*args, **kwargs)

    def __repr__(self):
        """a.__repr__(/)

        Return ``repr(self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return repr(self.__array__())

    def __rfloordiv__(self, lhs):
        """a.__rfloordiv__(value, /)

        Return ``value//self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import floor_divide

        return floor_divide(lhs, self)

    def __rmod__(self, lhs):
        """a.__rmod__(value, /)

        Return ``value%self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import remainder

        return remainder(lhs, self)

    def __rmul__(self, lhs):
        """a.__rmul__(value, /)

        Return ``value*self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import multiply

        return multiply(lhs, self)

    def __ror__(self, lhs):
        """a.__ror__(value, /)

        Return ``value|self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import logical_or

        return logical_or(lhs, self)

    def __rpow__(self, lhs):
        """__rpow__(value, /)

        Return ``pow(value, self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import power

        return power(lhs, self)

    def __rshift__(self, rhs):
        """a.__rshift__(value, /)

        Return ``self>>value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import right_shift

        return right_shift(self, rhs)

    def __rsub__(self, lhs):
        """a.__rsub__(value, /)

        Return ``value-self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import subtract

        return subtract(lhs, self)

    def __rtruediv__(self, lhs):
        """a.__rtruediv__(value, /)

        Return ``value/self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import true_divide

        return true_divide(lhs, self)

    def __rxor__(self, lhs):
        """a.__rxor__(value, /)

        Return ``value^self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import bitwise_xor

        return bitwise_xor(lhs, self)

    # __setattr__

    @add_boilerplate("value", mutates_self=True)
    def __setitem__(self, key, value):
        """__setitem__(key, value, /)

        Set ``self[key]=value``.

        """
        if key is None:
            raise KeyError("invalid key passed to cunumeric.ndarray")
        if value.dtype != self.dtype:
            temp = ndarray(value.shape, dtype=self.dtype, inputs=(value,))
            temp._thunk.convert(value._thunk)
            value = temp
        key = self._convert_key(key)
        self._thunk.set_item(key, value._thunk)

    def __setstate__(self, state):
        """a.__setstate__(state, /)

        For unpickling.

        The `state` argument must be a sequence that contains the following
        elements:

        Parameters
        ----------
        version : int
            optional pickle version. If omitted defaults to 0.
        shape : tuple
        dtype : data-type
        isFortran : bool
        rawdata : str or list
            a binary string with the data, or a list if 'a' is an object array

        """
        self.__array__().__setstate__(state)

    def __sizeof__(self, *args, **kwargs):
        return self.__array__().__sizeof__(*args, **kwargs)

    def __sub__(self, rhs):
        """a.__sub__(value, /)

        Return ``self-value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import subtract

        return subtract(self, rhs)

    def __str__(self):
        """a.__str__(/)

        Return ``str(self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return str(self.__array__())

    def __truediv__(self, rhs):
        """a.__truediv__(value, /)

        Return ``self/value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import true_divide

        return true_divide(self, rhs)

    def __xor__(self, rhs):
        """a.__xor__(value, /)

        Return ``self^value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import bitwise_xor

        return bitwise_xor(self, rhs)

    @add_boilerplate()
    def all(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        """a.all(axis=None, out=None, keepdims=False, initial=None, where=True)

        Returns True if all elements evaluate to True.

        Refer to :func:`cunumeric.all` for full documentation.

        See Also
        --------
        cunumeric.all : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self._perform_unary_reduction(
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
        """a.any(axis=None, out=None, keepdims=False, initial=None, where=True)

        Returns True if any of the elements of `a` evaluate to True.

        Refer to :func:`cunumeric.any` for full documentation.

        See Also
        --------
        cunumeric.any : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self._perform_unary_reduction(
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
        """a.argmax(axis=None, out=None)

        Return indices of the maximum values along the given axis.

        Refer to :func:`cunumeric.argmax` for full documentation.

        See Also
        --------
        cunumeric.argmax : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.size == 1:
            return 0
        if axis is None:
            axis = self.ndim - 1
        elif type(axis) != int:
            raise TypeError("'axis' argument for argmax must be an 'int'")
        elif axis < 0 or axis >= self.ndim:
            raise TypeError("invalid 'axis' argument for argmax " + str(axis))
        return self._perform_unary_reduction(
            UnaryRedCode.ARGMAX,
            self,
            axis=axis,
            dtype=np.dtype(np.int64),
            dst=out,
            check_types=False,
        )

    def argmin(self, axis=None, out=None):
        """a.argmin(axis=None, out=None)

        Return indices of the minimum values along the given axis.

        Refer to :func:`cunumeric.argmin` for detailed documentation.

        See Also
        --------
        cunumeric.argmin : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.size == 1:
            return 0
        if axis is None:
            axis = self.ndim - 1
        elif type(axis) != int:
            raise TypeError("'axis' argument for argmin must be an 'int'")
        elif axis < 0 or axis >= self.ndim:
            raise TypeError("invalid 'axis' argument for argmin " + str(axis))
        return self._perform_unary_reduction(
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
        """a.astype(dtype, order='C', casting='unsafe', subok=True, copy=True)

        Copy of the array, cast to a specified type.

        Parameters
        ----------
        dtype : str or data-type
            Typecode or data-type to which the array is cast.

        order : ``{'C', 'F', 'A', 'K'}``, optional
            Controls the memory layout order of the result.
            'C' means C order, 'F' means Fortran order, 'A'
            means 'F' order if all the arrays are Fortran contiguous,
            'C' order otherwise, and 'K' means as close to the
            order the array elements appear in memory as possible.
            Default is 'K'.

        casting : ``{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}``, optional
            Controls what kind of data casting may occur. Defaults to 'unsafe'
            for backwards compatibility.

            * 'no' means the data types should not be cast at all.
            * 'equiv' means only byte-order changes are allowed.
            * 'safe' means only casts which can preserve values are allowed.
            * 'same_kind' means only safe casts or casts within a kind,
                like float64 to float32, are allowed.
            * 'unsafe' means any data conversions may be done.

        subok : bool, optional
            If True, then sub-classes will be passed-through (default),
            otherwise the returned array will be forced to be a base-class
            array.

        copy : bool, optional
            By default, astype always returns a newly allocated array. If this
            is set to false, and the `dtype`, `order`, and `subok`
            requirements are satisfied, the input array is returned instead
            of a copy.

        Returns
        -------
        arr_t : ndarray
            Unless `copy` is False and the other conditions for returning the
            input array are satisfied (see description for `copy` input
            parameter), `arr_t` is a new array of the same shape as the input
            array, with dtype, order given by `dtype`, `order`.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
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
        """a.choose(choices, out=None, mode='raise')

        Use an index array to construct a new array from a set of choices.

        Refer to :func:`cunumeric.choose` for full documentation.

        See Also
        --------
        cunumeric.choose : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        a = self
        if out is not None:
            out = convert_to_cunumeric_ndarray(out)

        if isinstance(choices, list):
            choices = tuple(choices)
        is_tuple = isinstance(choices, tuple)
        if is_tuple:
            n = len(choices)
            dtypes = [ch.dtype for ch in choices]
            ch_dtype = np.find_common_type(dtypes, [])
            choices = tuple(
                convert_to_cunumeric_ndarray(choices[i]).astype(ch_dtype)
                for i in range(n)
            )

        else:
            choices = convert_to_cunumeric_ndarray(choices)
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

    def clip(self, min=None, max=None, out=None):
        """a.clip(min=None, max=None, out=None)

        Return an array whose values are limited to ``[min, max]``.

        One of max or min must be given.

        Refer to :func:`cunumeric.clip` for full documentation.

        See Also
        --------
        cunumeric.clip : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
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
                return convert_to_cunumeric_ndarray(out, share=True)
            else:
                return convert_to_cunumeric_ndarray(
                    self.__array__.clip(min, max)
                )
        return self._perform_unary_op(
            UnaryOpCode.CLIP, self, dst=out, extra_args=args
        )

    def conj(self):
        """a.conj()

        Complex-conjugate all elements.

        Refer to :func:`cunumeric.conjugate` for full documentation.

        See Also
        --------
        cunumeric.conjugate : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.dtype.kind == "c":
            result = self._thunk.conj()
            return ndarray(self.shape, dtype=self.dtype, thunk=result)
        else:
            return self

    def conjugate(self):
        """a.conjugate()

        Return the complex conjugate, element-wise.

        Refer to :func:`cunumeric.conjugate` for full documentation.

        See Also
        --------
        cunumeric.conjugate : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.conj()

    def copy(self, order="C"):
        """copy()

        Get a copy of the iterator as a 1-D array.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # We don't care about dimension order in cuNumeric
        return self.__copy__()

    # diagonal helper. Will return diagonal for arbitrary number of axes;
    # currently offset option is implemented only for the case of number of
    # axes=2. This restriction can be lifted in the future if there is a
    # use case of having arbitrary number of offsets
    def _diag_helper(
        self,
        offset=0,
        axes=None,
        extract=True,
        trace=False,
        out=None,
        dtype=None,
    ):
        # _diag_helper can be used only for arrays with dim>=1
        if self.ndim < 1:
            raise ValueError("_diag_helper is implemented for dim>=1")
        # out should be passed only for Trace
        if out is not None and not trace:
            raise ValueError("_diag_helper supports out only for trace=True")
        # dtype should be passed only for Trace
        if dtype is not None and not trace:
            raise ValueError("_diag_helper supports dtype only for trace=True")

        elif self.ndim == 1:
            if axes is not None:
                raise ValueError(
                    "Axes shouldn't be specified when getting "
                    "diagonal for 1D array"
                )
            m = self.shape[0] + np.abs(offset)
            out = ndarray((m, m), dtype=self.dtype, inputs=(self,))
            diag_size = self.shape[0]
            out._thunk._diag_helper(
                self._thunk, offset=offset, naxes=0, extract=False, trace=False
            )
        else:
            N = len(axes)
            if len(axes) != len(set(axes)):
                raise ValueError(
                    "axes passed to _diag_helper should be all different"
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
                    "number of axes passed to the _diag_helper"
                    " should be more than 1"
                )

            tr_shape = tuple(a.shape[i] for i in range(a.ndim - N))
            # calculate shape of the output array
            if trace:
                if N != 2:
                    raise ValueError(
                        " exactly 2 axes should be passed to trace"
                    )
                if self.ndim == 2:
                    out_shape = (1,)
                elif self.ndim > 2:
                    out_shape = tr_shape
                else:
                    raise ValueError(
                        "dimension of the array for trace operation:"
                        " should be >=2"
                    )
            else:
                out_shape = tr_shape + (diag_size,)

            if out is not None:
                if out.shape != out_shape:
                    raise ValueError("output array has wrong shape")
                a = a._maybe_convert(out.dtype, (a, out))
            else:
                out = ndarray(
                    shape=out_shape, dtype=self.dtype, inputs=(self,)
                )
            if out is None and dtype is not None:
                a = a._maybe_convert(dtype, (a,))

            out._thunk._diag_helper(
                a._thunk, offset=offset, naxes=N, extract=extract, trace=trace
            )
        return out

    def diagonal(
        self, offset=0, axis1=None, axis2=None, extract=True, axes=None
    ):
        """a.diagonal(offset=0, axis1=None, axis2=None)

        Return specified diagonals.

        Refer to :func:`cunumeric.diagonal` for full documentation.

        See Also
        --------
        cunumeric.diagonal : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
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
        return self._diag_helper(offset=offset, axes=axes, extract=extract)

    @add_boilerplate()
    def trace(self, offset=0, axis1=None, axis2=None, dtype=None, out=None):
        """a.trace(offset=0, axis1=None, axis2=None, dtype = None, out = None)

        Return the sum along diagonals of the array.

        Refer to :func:`cunumeric.trace` for full documentation.

        See Also
        --------
        cunumeric.trace : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.ndim < 2:
            raise ValueError(
                "trace operation can't be called on a array with DIM<2"
            )

        axes = []
        if (axis1 is None) and (axis2 is None):
            # default values for axis
            axes = (0, 1)
        elif (axis1 is None) or (axis2 is None):
            raise ValueError("both axes should be passed")
        else:
            axes = (axis1, axis2)

        res = self._diag_helper(
            offset=offset, axes=axes, trace=True, out=out, dtype=dtype
        )

        # for 2D arrays we must return scalar
        if self.ndim == 2:
            res = res[0]

        return res

    @add_boilerplate("rhs")
    def dot(self, rhs, out=None):
        """a.dot(rhs, out=None)

        Return the dot product of this array with ``rhs``.

        Refer to :func:`cunumeric.dot` for full documentation.

        See Also
        --------
        cunumeric.dot : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from .module import _contract  # work around circular import

        if self.ndim == 0 or rhs.ndim == 0:
            from ._ufunc import multiply

            return multiply(self, rhs, out=out)

        (self_modes, rhs_modes, out_modes) = dot_modes(self.ndim, rhs.ndim)
        return _contract(
            self_modes,
            rhs_modes,
            out_modes,
            self,
            rhs,
            out=out,
        )

    def dump(self, file):
        """a.dump(file)

        Dump a pickle of the array to the specified file.

        The array can be read back with pickle.load or cunumeric.load.

        Parameters
        ----------
        file : str or `pathlib.Path`
            A string naming the dump file.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        self.__array__().dump(file=file)

    def dumps(self):
        """a.dumps()

        Returns the pickle of the array as a string.

        pickle.loads will convert the string back to an array.

        Parameters
        ----------
        None

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__array__().dumps()

    def _normalize_axes_shape(self, axes, s):
        user_axes = axes is not None
        user_sizes = s is not None
        if user_axes and user_sizes and len(axes) != len(s):
            raise ValueError("Shape and axes have different lengths")
        if user_axes:
            fft_axes = [ax if ax >= 0 else ax + self.ndim for ax in axes]
            if min(fft_axes) < 0 or max(fft_axes) >= self.ndim:
                raise ValueError(
                    "Axis is out of bounds for array of size {}".format(
                        self.ndim
                    )
                )
        else:
            fft_axes = range(len(s)) if user_sizes else range(self.ndim)

        fft_s = list(self.shape)
        if user_sizes:
            for idx, ax in enumerate(fft_axes):
                fft_s[ax] = s[idx]
        return np.asarray(fft_axes), np.asarray(fft_s)

    def fft(self, s, axes, kind, direction, norm):
        """a.fft(s, axes, kind, direction, norm)

        Return the ``kind`` ``direction`` FFT of this array
        with normalization ``norm``.

        Common entrypoint for FFT functionality in cunumeric.fft module.

        See Also
        --------
        cunumeric.fft : FFT functions for different ``kind`` and
        ``direction`` arguments

        Availability
        --------
        Single GPU

        """
        # Dimensions check
        if self.ndim > 3:
            raise NotImplementedError(
                f"{self.ndim}-D arrays are not supported yet"
            )

        # Type
        fft_output_type = kind.output_dtype

        # Axes and sizes
        user_sizes = s is not None
        fft_axes, fft_s = self._normalize_axes_shape(axes, s)

        # Shape
        fft_input = self
        fft_input_shape = np.asarray(self.shape)
        fft_output_shape = np.asarray(self.shape)
        if user_sizes:
            # Zero padding if any of the user sizes is larger than input
            zeropad_input = self
            if np.any(np.greater(fft_s, fft_input_shape)):
                # Create array with superset shape, fill with zeros,
                # and copy input in
                max_size = tuple(np.maximum(fft_s, fft_input_shape))
                zeropad_input = ndarray(shape=max_size, dtype=fft_input.dtype)
                zeropad_input.fill(0)
                slices = tuple(slice(0, i) for i in fft_input.shape)
                zeropad_input._thunk.set_item(slices, fft_input._thunk)

            # Slicing according to final shape
            for idx, ax in enumerate(fft_axes):
                fft_input_shape[ax] = s[idx]
            # TODO: always copying is not the best idea,
            # sometimes a view of the original input will do
            slices = tuple(slice(0, i) for i in fft_s)
            fft_input = ndarray(
                shape=fft_input_shape,
                thunk=zeropad_input._thunk.get_item(slices),
            )
            fft_output_shape = np.copy(fft_input_shape)

        # R2C/C2R require different output shapes
        if fft_output_type != self.dtype:
            # R2C/C2R dimension is the last axis
            lax = fft_axes[-1]
            if direction == FFTDirection.FORWARD:
                fft_output_shape[lax] = fft_output_shape[lax] // 2 + 1
            else:
                if user_sizes:
                    fft_output_shape[lax] = s[-1]
                else:
                    fft_output_shape[lax] = 2 * (fft_input.shape[lax] - 1)

        # Execute FFT backend
        out = ndarray(
            shape=fft_output_shape,
            dtype=fft_output_type,
        )
        fft_input._thunk.fft(out._thunk, fft_axes, kind, direction)

        # Normalization
        fft_norm = FFTNormalization.from_string(norm)
        do_normalization = any(
            (
                fft_norm == FFTNormalization.ORTHOGONAL,
                fft_norm == FFTNormalization.FORWARD
                and direction == FFTDirection.FORWARD,
                fft_norm == FFTNormalization.INVERSE
                and direction == FFTDirection.INVERSE,
            )
        )
        if do_normalization:
            if direction == FFTDirection.FORWARD:
                norm_shape = fft_input.shape
            else:
                norm_shape = out.shape
            norm_shape_along_axes = [norm_shape[ax] for ax in fft_axes]
            factor = np.product(norm_shape_along_axes)
            if fft_norm == FFTNormalization.ORTHOGONAL:
                factor = np.sqrt(factor)
            return out / factor

        return out

    def fill(self, value):
        """a.fill(value)

        Fill the array with a scalar value.

        Parameters
        ----------
        value : scalar
            All elements of `a` will be assigned this value.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        val = np.array(value, dtype=self.dtype)
        self._thunk.fill(val)

    def flatten(self, order="C"):
        """a.flatten(order='C')

        Return a copy of the array collapsed into one dimension.

        Parameters
        ----------
        order : ``{'C', 'F', 'A', 'K'}``, optional
            'C' means to flatten in row-major (C-style) order.
            'F' means to flatten in column-major (Fortran-
            style) order. 'A' means to flatten in column-major
            order if `a` is Fortran *contiguous* in memory,
            row-major order otherwise. 'K' means to flatten
            `a` in the order the elements occur in memory.
            The default is 'C'.

        Returns
        -------
        y : ndarray
            A copy of the input array, flattened to one dimension.

        See Also
        --------
        ravel : Return a flattened array.
        flat : A 1-D flat iterator over the array.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # Same as 'ravel' because cuNumeric creates a new array by 'reshape'
        return self.reshape(-1, order=order)

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
        """a.item(*args)

        Copy an element of an array to a standard Python scalar and return it.

        Parameters
        ----------
        \\*args :

            * none: in this case, the method only works for arrays
                with one element (`a.size == 1`), which element is
                copied into a standard Python scalar object and returned.
            * int_type: this argument is interpreted as a flat index into
                the array, specifying which element to copy and return.
            * tuple of int_types: functions as does a single int_type
                argument, except that the argument is interpreted as an
                nd-index into the array.

        Returns
        -------
        z : scalar
            A copy of the specified element of the array as a suitable
            Python scalar

        Notes
        -----
        When the data type of `a` is longdouble or clongdouble, item() returns
        a scalar array object because there is no available Python scalar that
        would not lose information. Void arrays return a buffer object for
        item(), unless fields are defined, in which case a tuple is returned.
        `item` is very similar to a[args], except, instead of an array scalar,
        a standard Python scalar is returned. This can be useful for speeding
        up access to elements of the array and doing arithmetic on elements of
        the array using Python's optimized math.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        key = self._convert_singleton_key(args)
        result = self[key]
        assert result.shape == ()
        return result._thunk.__numpy_array__()

    def itemset(self, *args):
        """a.itemset(*args)

        Insert scalar into an array (scalar is cast to array's dtype,
        if possible)

        There must be at least 1 argument, and define the last argument
        as *item*.  Then, ``a.itemset(*args)`` is equivalent to but faster
        than ``a[args] = item``.  The item should be a scalar value and `args`
        must select a single item in the array `a`.

        Parameters
        ----------
        \\*args :
            If one argument: a scalar, only used in case `a` is of size 1.
            If two arguments: the last argument is the value to be set
            and must be a scalar, the first argument specifies a single array
            element location. It is either an int or a tuple.

        Notes
        -----
        Compared to indexing syntax, `itemset` provides some speed increase
        for placing a scalar into a particular location in an `ndarray`,
        if you must do this.  However, generally this is discouraged:
        among other problems, it complicates the appearance of the code.
        Also, when using `itemset` (and `item`) inside a loop, be sure
        to assign the methods to a local variable to avoid the attribute
        look-up at each loop iteration.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
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
        """a.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

        Return the maximum along a given axis.

        Refer to :func:`cunumeric.amax` for full documentation.

        See Also
        --------
        cunumeric.amax : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self._perform_unary_reduction(
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
        """a.mean(axis=None, dtype=None, out=None, keepdims=False)

        Returns the average of the array elements along given axis.

        Refer to :func:`cunumeric.mean` for full documentation.

        See Also
        --------
        cunumeric.mean : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if axis is not None and type(axis) != int:
            raise NotImplementedError(
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
            sum_array.__itruediv__(
                np.array(divisor, dtype=sum_array.dtype),
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
        """a.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

        Return the minimum along a given axis.

        Refer to :func:`cunumeric.amin` for full documentation.

        See Also
        --------
        cunumeric.amin : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self._perform_unary_reduction(
            UnaryRedCode.MIN,
            self,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    @add_boilerplate()
    def partition(self, kth, axis=-1, kind="introselect", order=None):
        """a.partition(kth, axis=-1, kind='introselect', order=None)

        Partition of an array in-place.

        Refer to :func:`cunumeric.partition` for full documentation.

        See Also
        --------
        cunumeric.partition : equivalent function

        Availability
        --------
        Multiple GPUs, Single CPU

        """
        self._thunk.partition(
            rhs=self._thunk, kth=kth, axis=axis, kind=kind, order=order
        )

    @add_boilerplate()
    def argpartition(self, kth, axis=-1, kind="introselect", order=None):
        """a.argpartition(kth, axis=-1, kind='introselect', order=None)

        Returns the indices that would partition this array.

        Refer to :func:`cunumeric.argpartition` for full documentation.

        See Also
        --------
        cunumeric.argpartition : equivalent function

        Availability
        --------
        Multiple GPUs, Single CPU

        """
        result = ndarray(self.shape, np.int64)
        result._thunk.partition(
            rhs=self._thunk,
            argpartition=True,
            kth=kth,
            axis=axis,
            kind=kind,
            order=order,
        )
        return result

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
        """a.prod(axis=None, dtype=None, out=None, keepdims=False, initial=1,
        where=True)

        Return the product of the array elements over the given axis

        Refer to :func:`cunumeric.prod` for full documentation.

        See Also
        --------
        cunumeric.prod : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
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
        return self._perform_unary_reduction(
            UnaryRedCode.PROD,
            self_array,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def ravel(self, order="C"):
        """a.ravel(order="C")

        Return a flattened array.

        Refer to :func:`cunumeric.ravel` for full documentation.

        See Also
        --------
        cunumeric.ravel : equivalent function
        ndarray.flat : a flat iterator on the array.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.reshape(-1, order=order)

    def reshape(self, shape, order="C"):
        """a.reshape(shape, order='C')

        Returns an array containing the same data with a new shape.

        Refer to :func:`cunumeric.reshape` for full documentation.

        See Also
        --------
        cunumeric.reshape : equivalent function


        Availability
        --------
        Multiple GPUs, Multiple CPUs
        """
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
        """a.setflags(write=None, align=None, uic=None)

        Set array flags WRITEABLE, ALIGNED, WRITEBACKIFCOPY,
        respectively.

        These Boolean-valued flags affect how numpy interprets the memory
        area used by `a` (see Notes below). The ALIGNED flag can only
        be set to True if the data is actually aligned according to the type.
        The WRITEBACKIFCOPY and flag can never be set
        to True. The flag WRITEABLE can only be set to True if the array owns
        its own memory, or the ultimate owner of the memory exposes a
        writeable buffer interface, or is a string. (The exception for string
        is made so that unpickling can be done without copying memory.)

        Parameters
        ----------
        write : bool, optional
            Describes whether or not `a` can be written to.
        align : bool, optional
            Describes whether or not `a` is aligned properly for its type.
        uic : bool, optional
            Describes whether or not `a` is a copy of another "base" array.

        Notes
        -----
        Array flags provide information about how the memory area used
        for the array is to be interpreted. There are 7 Boolean flags
        in use, only four of which can be changed by the user:
        WRITEBACKIFCOPY, WRITEABLE, and ALIGNED.

        WRITEABLE (W) the data area can be written to;

        ALIGNED (A) the data and strides are aligned appropriately for the
        hardware (as determined by the compiler);

        WRITEBACKIFCOPY (X) this array is a copy of some other array
        (referenced by .base). When the C-API function
        PyArray_ResolveWritebackIfCopy is called, the base array will be
        updated with the contents of this array.

        All flags can be accessed using the single (upper case) letter as well
        as the full name.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        self.__array__().setflags(write=write, align=align, uic=uic)

    def sort(self, axis=-1, kind="quicksort", order=None):
        """a.sort(axis=-1, kind=None, order=None)

        Sort an array in-place.

        Refer to :func:`cunumeric.sort` for full documentation.

        See Also
        --------
        cunumeric.sort : equivalent function

        Availability
        --------
        Multiple GPUs, Single CPU

        """
        self._thunk.sort(rhs=self._thunk, axis=axis, kind=kind, order=order)

    def argsort(self, axis=-1, kind=None, order=None):
        """a.argsort(axis=-1, kind=None, order=None)

        Returns the indices that would sort this array.

        Refer to :func:`cunumeric.argsort` for full documentation.

        See Also
        --------
        cunumeric.argsort : equivalent function

        Availability
        --------
        Multiple GPUs, Single CPU

        """
        result = ndarray(self.shape, np.int64)
        result._thunk.sort(
            rhs=self._thunk, argsort=True, axis=axis, kind=kind, order=order
        )
        return result

    def squeeze(self, axis=None):
        """a.squeeze(axis=None)

        Remove axes of length one from `a`.

        Refer to :func:`cunumeric.squeeze` for full documentation.

        See Also
        --------
        cunumeric.squeeze : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
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
        """a.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0,
        where=True)

        Return the sum of the array elements over the given axis.

        Refer to :func:`cunumeric.sum` for full documentation.

        See Also
        --------
        cunumeric.sum : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
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
        return self._perform_unary_reduction(
            UnaryRedCode.SUM,
            self_array,
            axis=axis,
            dst=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def swapaxes(self, axis1, axis2):
        """a.swapaxes(axis1, axis2)

        Return a view of the array with `axis1` and `axis2` interchanged.

        Refer to :func:`cunumeric.swapaxes` for full documentation.

        See Also
        --------
        cunumeric.swapaxes : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
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
        """a.tofile(fid, sep="", format="%s")

        Write array to a file as text or binary (default).

        Data is always written in 'C' order, independent of the order of `a`.
        The data produced by this method can be recovered using the function
        fromfile().

        Parameters
        ----------
        fid : ``file`` or str or pathlib.Path
            An open file object, or a string containing a filename.
        sep : str
            Separator between array items for text output.
            If "" (empty), a binary file is written, equivalent to
            ``file.write(a.tobytes())``.
        format : str
            Format string for text file output.
            Each entry in the array is formatted to text by first converting
            it to the closest Python type, and then using "format" % item.

        Notes
        -----
        This is a convenience function for quick storage of array data.
        Information on endianness and precision is lost, so this method is not
        a good choice for files intended to archive data or transport data
        between machines with different endianness. Some of these problems can
        be overcome by outputting the data as text files, at the expense of
        speed and file size.

        When fid is a file object, array contents are directly written to the
        file, bypassing the file object's ``write`` method. As a result,
        tofile cannot be used with files objects supporting compression (e.g.,
        GzipFile) or file-like objects that do not support ``fileno()`` (e.g.,
        BytesIO).

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__array__().tofile(fid=fid, sep=sep, format=format)

    def tobytes(self, order="C"):
        """a.tobytes(order='C')

        Construct Python bytes containing the raw data bytes in the array.

        Constructs Python bytes showing a copy of the raw contents of
        data memory. The bytes object is produced in C-order by default.

        This behavior is controlled by the ``order`` parameter.

        Parameters
        ----------
        order : ``{'C', 'F', 'A'}``, optional
            Controls the memory layout of the bytes object. 'C' means C-order,
            'F' means F-order, 'A' (short for *Any*) means 'F' if `a` is
            Fortran contiguous, 'C' otherwise. Default is 'C'.

        Returns
        -------
        s : bytes
            Python bytes exhibiting a copy of `a`'s raw data.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__array__().tobytes(order=order)

    def tolist(self):
        """a.tolist()

        Return the array as an ``a.ndim``-levels deep nested list of Python
        scalars.

        Return a copy of the array data as a (nested) Python list.
        Data items are converted to the nearest compatible builtin Python
        type, via the `~cunumeric.ndarray.item` function.

        If ``a.ndim`` is 0, then since the depth of the nested list is 0, it
        will not be a list at all, but a simple Python scalar.

        Parameters
        ----------
        None

        Returns
        -------
        y : Any
            The possibly nested list of array elements. (object, or list of
            object, or list of list of object, or ...)

        Notes
        -----
        The array may be recreated via ``a = cunumeric.array(a.tolist())``,
        although this may sometimes lose precision.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__array__().tolist()

    def tostring(self, order="C"):
        """a.tostring(order='C')

        A compatibility alias for `tobytes`, with exactly the same behavior.
        Despite its name, it returns `bytes` not `str`.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__array__().tostring(order=order)

    def transpose(self, axes=None):
        """a.transpose(axes=None)

        Returns a view of the array with axes transposed.

        For a 1-D array this has no effect, as a transposed vector is simply
        the same vector. To convert a 1-D array into a 2D column vector, an
        additional dimension must be added. `np.atleast2d(a).T` achieves this,
        as does `a[:, np.newaxis]`.

        For a 2-D array, this is a standard matrix transpose.

        For an n-D array, if axes are given, their order indicates how the
        axes are permuted (see Examples). If axes are not provided and
        ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
        ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

        Parameters
        ----------
        axes : None or tuple[int]

            * None or no argument: reverses the order of the axes.
            * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
                `i`-th axis becomes `a.transpose()`'s `j`-th axis.

        Returns
        -------
        out : ndarray
            View of `a`, with axes suitably permuted.

        See Also
        --------
        transpose : Equivalent function
        ndarray.T : Array property returning the array transposed.
        ndarray.reshape : Give a new shape to an array without changing its
            data.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
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
        """
        Reverse the order of elements in an array along the given axis.

        The shape of the array is preserved, but the elements are reordered.

        Parameters
        ----------
        axis : None or int or tuple[int], optional
            Axis or axes along which to flip over. The default, axis=None, will
            flip over all of the axes of the input array.  If axis is negative
            it counts from the last to the first axis.

            If axis is a tuple of ints, flipping is performed on all of the
            axes specified in the tuple.

        Returns
        -------
        out : array_like
            A view of `m` with the entries of axis reversed.  Since a view is
            returned, this operation is done in constant time.

        Availability
        --------
        Single GPU, Single CPU

        """
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

    def unique(self):
        """a.unique()

        Find the unique elements of an array.

        Refer to :func:`cunumeric.unique` for full documentation.

        See Also
        --------
        cunumeric.unique : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        thunk = self._thunk.unique()
        return ndarray(shape=thunk.shape, thunk=thunk)

    @classmethod
    def _get_where_thunk(cls, where, out_shape):
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
        """Determine common type following standard coercion rules.

        Parameters
        ----------
        \\*args :
            A list of dtypes or dtype convertible objects representing arrays
            or scalars.


        Returns
        -------
        datatype : data-type
            The common data type, which is the maximum of the array types,
            ignoring any scalar types , unless the maximum scalar type is of a
            different kind (`dtype.kind`). If the kind is not understood, then
            None is returned.

        """
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
    def _perform_unary_op(
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
                    src._thunk,
                    cls._get_where_thunk(where, dst.shape),
                    extra_args,
                )
                dst._thunk.convert(temp._thunk)
            else:
                dst._thunk.unary_op(
                    op,
                    src._thunk,
                    cls._get_where_thunk(where, dst.shape),
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
                    src._thunk,
                    cls._get_where_thunk(where, dst.shape),
                    extra_args,
                )
                dst._thunk.convert(temp._thunk)
            else:
                dst._thunk.unary_op(
                    op,
                    src._thunk,
                    cls._get_where_thunk(where, dst.shape),
                    extra_args,
                )
        return dst

    # For performing reduction unary operations
    @classmethod
    def _perform_unary_reduction(
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
        if (
            op
            in (
                UnaryRedCode.ARGMAX,
                UnaryRedCode.ARGMIN,
                UnaryRedCode.MAX,
                UnaryRedCode.MIN,
            )
            and src.dtype.kind == "c"
        ):
            raise NotImplementedError(
                "(arg)max/min not supported for complex-type arrays"
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
                    cls._get_where_thunk(where, dst.shape),
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
                    cls._get_where_thunk(where, dst.shape),
                    axes,
                    keepdims,
                    args,
                    initial,
                )
        else:
            dst._thunk.unary_reduction(
                op,
                src._thunk,
                cls._get_where_thunk(where, dst.shape),
                axes,
                keepdims,
                args,
                initial,
            )
        return dst

    @classmethod
    def _perform_binary_reduction(
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
    def _perform_where(cls, mask, one, two):
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
