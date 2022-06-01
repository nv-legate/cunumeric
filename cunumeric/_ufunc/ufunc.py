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

from typing import TYPE_CHECKING, Any, Literal, Sequence, Union

import numpy as np

from ..array import convert_to_cunumeric_ndarray, ndarray
from ..config import BinaryOpCode, UnaryOpCode, UnaryRedCode

if TYPE_CHECKING:
    import numpy.typing as npt

CastingKind = Union[
    Literal["no", "equiv", "safe", "same_kind", "unsafe"], None
]

_UNARY_DOCSTRING_TEMPLATE = """{}

Parameters
----------
x : array_like
    Input array.
out : ndarray, or None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
out : ndarray or scalar
    Result.
    This is a scalar if `x` is a scalar.

See Also
--------
numpy.{}

Availability
--------
Multiple GPUs, Multiple CPUs
"""

_MULTIOUT_UNARY_DOCSTRING_TEMPLATE = """{}

Parameters
----------
x : array_like
    Input array.
out : tuple[ndarray or None], or None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
y1 : ndarray
    This is a scalar if `x` is a scalar.
y2 : ndarray
    This is a scalar if `x` is a scalar.

See Also
--------
numpy.{}

Availability
--------
Multiple GPUs, Multiple CPUs
"""

_BINARY_DOCSTRING_TEMPLATE = """{}

Parameters
----------
x1, x2 : array_like
    Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable
    to a common shape (which becomes the shape of the output).
out : ndarray, None, or tuple[ndarray or None], optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
y : ndarray or scalar
    Result.
    This is a scalar if both `x1` and `x2` are scalars.

See Also
--------
numpy.{}

Availability
--------
Multiple GPUs, Multiple CPUs
"""

float_dtypes = ["e", "f", "d"]

complex_dtypes = ["F", "D"]

float_and_complex = float_dtypes + complex_dtypes

integer_dtypes = [
    "b",
    "B",
    "h",
    "H",
    "i",
    "I",
    "l",
    "L",
    "q",
    "Q",
]

all_but_boolean = integer_dtypes + float_and_complex

all_dtypes = ["?"] + all_but_boolean


def predicate_types_of(dtypes: Sequence[str]) -> list[str]:
    return [ty + "?" for ty in dtypes]


def relation_types_of(dtypes: Sequence[str]) -> list[str]:
    return [ty * 2 + "?" for ty in dtypes]


def to_dtypes(chars: str) -> tuple[np.dtype[Any], ...]:
    return tuple(np.dtype(char) for char in chars)


class ufunc:
    def __init__(self, name: str, doc: str, types: dict[Any, str]) -> None:
        self._name = name
        self._types = types
        self.__doc__ = doc

        assert len(self._types)
        in_ty, out_ty = next(iter(self._types.items()))
        self._nin = len(in_ty)
        self._nout = len(out_ty)

    @property
    def nin(self) -> int:
        return self._nin

    @property
    def nout(self) -> int:
        return self._nout

    @property
    def types(self) -> list[str]:
        return [
            f"{''.join(in_tys)}->{''.join(out_tys)}"
            for in_tys, out_tys in self._types.items()
        ]

    @property
    def ntypes(self) -> int:
        return len(self._types)

    def _maybe_cast_input(
        self, arr: ndarray, to_dtype: np.dtype[Any], casting: CastingKind
    ) -> ndarray:
        if arr.dtype == to_dtype:
            return arr

        if not np.can_cast(arr.dtype, to_dtype, casting=casting):
            raise TypeError(
                f"Cannot cast ufunc '{self._name}' input from "
                f"{arr.dtype} to {to_dtype} with casting rule '{casting}'"
            )

        return arr.astype(to_dtype)

    def _maybe_create_result(
        self,
        out: Union[ndarray, None],
        out_shape: tuple[int, ...],
        res_dtype: np.dtype[Any],
        casting: CastingKind,
        inputs: Union[tuple[ndarray, bool], tuple[ndarray, ndarray, bool]],
    ) -> ndarray:
        if out is None:
            return ndarray(shape=out_shape, dtype=res_dtype, inputs=inputs)
        elif out.dtype != res_dtype:
            if not np.can_cast(res_dtype, out.dtype, casting=casting):
                raise TypeError(
                    f"Cannot cast ufunc '{self._name}' output from "
                    f"{res_dtype} to {out.dtype} with casting rule "
                    f"'{casting}'"
                )
            return ndarray(shape=out.shape, dtype=res_dtype, inputs=inputs)
        else:
            return out

    @staticmethod
    def _maybe_cast_output(
        out: Union[ndarray, None], result: ndarray
    ) -> ndarray:
        if out is None or out is result:
            return result
        out._thunk.convert(result._thunk, warn=False)
        return out

    @staticmethod
    def _maybe_convert_output_to_cunumeric_ndarray(
        out: Union[ndarray, npt.NDArray[Any], None]
    ) -> Union[ndarray, None]:
        if out is None:
            return None
        if isinstance(out, ndarray):
            return out
        if isinstance(out, np.ndarray):
            return convert_to_cunumeric_ndarray(out, share=True)
        raise TypeError("return arrays must be of ArrayType")

    def _prepare_operands(
        self, *args: Any, out: Union[ndarray, None], where: bool = True
    ) -> tuple[
        Sequence[ndarray],
        Sequence[Union[ndarray, None]],
        tuple[int, ...],
        bool,
    ]:
        max_nargs = self.nin + self.nout
        if len(args) < self.nin or len(args) > max_nargs:
            raise TypeError(
                f"{self._name}() takes from {self.nin} to {max_nargs} "
                f"positional arguments but {len(args)} were given"
            )

        inputs = tuple(
            convert_to_cunumeric_ndarray(arr) for arr in args[: self.nin]
        )

        if len(args) > self.nin:
            if out is not None:
                raise TypeError(
                    "cannot specify 'out' as both a positional and keyword "
                    "argument"
                )
            computed_out = args[self.nin :]
            # Missing outputs are treated as Nones
            computed_out += (None,) * (self.nout - len(computed_out))
        elif out is None:
            computed_out = (None,) * self.nout
        elif not isinstance(out, tuple):
            computed_out = (out,)

        outputs = tuple(
            self._maybe_convert_output_to_cunumeric_ndarray(arr)
            for arr in computed_out
        )

        if self.nout != len(outputs):
            raise ValueError(
                "The 'out' tuple must have exactly one entry "
                "per ufunc output"
            )

        shapes = [arr.shape for arr in inputs]
        shapes.extend(arr.shape for arr in outputs if arr is not None)

        # Check if the broadcasting is possible
        out_shape = np.broadcast_shapes(*shapes)

        for out in outputs:
            if out is not None and out.shape != out_shape:
                raise ValueError(
                    f"non-broadcastable output operand with shape "
                    f"{out.shape} doesn't match the broadcast shape "
                    f"{out_shape}"
                )

        if not isinstance(where, bool) or not where:
            raise NotImplementedError(
                "the 'where' keyword is not yet supported"
            )

        return inputs, outputs, out_shape, where

    def __repr__(self) -> str:
        return f"<ufunc {self._name}>"


class unary_ufunc(ufunc):
    def __init__(
        self,
        name: str,
        doc: str,
        op_code: UnaryOpCode,
        types: dict[str, str],
        overrides: dict[str, UnaryOpCode],
    ) -> None:
        super().__init__(name, doc, types)
        self._op_code = op_code
        self._resolution_cache: dict[np.dtype[Any], np.dtype[Any]] = {}
        self._overrides = overrides

    def _resolve_dtype(
        self, arr: ndarray, precision_fixed: bool
    ) -> tuple[ndarray, np.dtype[Any]]:
        if arr.dtype.char in self._types:
            return arr, np.dtype(self._types[arr.dtype.char])

        if not precision_fixed:
            if arr.dtype in self._resolution_cache:
                to_dtype = self._resolution_cache[arr.dtype]
                arr = arr.astype(to_dtype)
                return arr, np.dtype(self._types[to_dtype.char])

        chosen = None
        if not precision_fixed:
            for in_ty in self._types.keys():
                if np.can_cast(arr.dtype, in_ty):
                    chosen = in_ty
                    break

        if chosen is None:
            raise TypeError(
                f"No matching signature of ufunc {self._name} is found "
                "for the given casting"
            )

        to_dtype = np.dtype(chosen)
        self._resolution_cache[arr.dtype] = to_dtype

        return arr.astype(to_dtype), np.dtype(self._types[to_dtype.char])

    def __call__(
        self,
        *args: Any,
        out: Union[ndarray, None] = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: Union[np.dtype[Any], None] = None,
        **kwargs: Any,
    ) -> ndarray:
        (x,), (out,), out_shape, where = self._prepare_operands(
            *args, out=out, where=where
        )

        # If no dtype is given to prescribe the accuracy, we use the dtype
        # of the input
        precision_fixed = False
        if dtype is not None:
            # If a dtype is given, that determines the precision
            # of the computation.
            precision_fixed = True
            x = self._maybe_cast_input(x, dtype, casting)

        # Resolve the dtype to use for the computation and cast the input
        # if necessary. If the dtype is already fixed by the caller,
        # the dtype must be one of the dtypes supported by this operation.
        x, res_dtype = self._resolve_dtype(x, precision_fixed)

        result = self._maybe_create_result(
            out, out_shape, res_dtype, casting, (x, where)
        )

        op_code = self._overrides.get(x.dtype.char, self._op_code)
        result._thunk.unary_op(op_code, x._thunk, where, ())

        return self._maybe_cast_output(out, result)


class multiout_unary_ufunc(ufunc):
    def __init__(
        self, name: str, doc: str, op_code: UnaryOpCode, types: dict[Any, Any]
    ) -> None:
        super().__init__(name, doc, types)
        self._op_code = op_code
        self._resolution_cache: dict[np.dtype[Any], np.dtype[Any]] = {}

    def _resolve_dtype(
        self, arr: ndarray, precision_fixed: bool
    ) -> tuple[ndarray, tuple[np.dtype[Any], ...]]:
        if arr.dtype.char in self._types:
            return arr, to_dtypes(self._types[arr.dtype.char])

        if not precision_fixed:
            if arr.dtype in self._resolution_cache:
                to_dtype = self._resolution_cache[arr.dtype]
                arr = arr.astype(to_dtype)
                return arr, to_dtypes(self._types[to_dtype.char])

        chosen = None
        if not precision_fixed:
            for in_ty in self._types.keys():
                if np.can_cast(arr.dtype, in_ty):
                    chosen = in_ty
                    break

        if chosen is None:
            raise TypeError(
                f"No matching signature of ufunc {self._name} is found "
                "for the given casting"
            )

        to_dtype = np.dtype(chosen)
        self._resolution_cache[arr.dtype] = to_dtype

        return arr.astype(to_dtype), to_dtypes(self._types[to_dtype.char])

    def __call__(
        self,
        *args: Any,
        out: Union[ndarray, None] = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: Union[np.dtype[Any], None] = None,
        **kwargs: Any,
    ) -> tuple[ndarray, ...]:
        (x,), outs, out_shape, where = self._prepare_operands(
            *args, out=out, where=where
        )

        # If no dtype is given to prescribe the accuracy, we use the dtype
        # of the input
        precision_fixed = False
        if dtype is not None:
            # If a dtype is given, that determines the precision
            # of the computation.
            precision_fixed = True
            x = self._maybe_cast_input(x, dtype, casting)

        # Resolve the dtype to use for the computation and cast the input
        # if necessary. If the dtype is already fixed by the caller,
        # the dtype must be one of the dtypes supported by this operation.
        x, res_dtypes = self._resolve_dtype(x, precision_fixed)

        results = tuple(
            self._maybe_create_result(
                out, out_shape, res_dtype, casting, (x, where)
            )
            for out, res_dtype in zip(outs, res_dtypes)
        )

        result_thunks = tuple(result._thunk for result in results)
        result_thunks[0].unary_op(
            self._op_code, x._thunk, where, (), multiout=result_thunks[1:]
        )

        return tuple(
            self._maybe_cast_output(out, result)
            for out, result in zip(outs, results)
        )


class binary_ufunc(ufunc):
    def __init__(
        self,
        name: str,
        doc: str,
        op_code: BinaryOpCode,
        types: dict[tuple[str, str], str],
        red_code: Union[UnaryRedCode, None] = None,
        use_common_type: bool = True,
    ) -> None:
        super().__init__(name, doc, types)
        self._op_code = op_code
        self._resolution_cache: dict[
            tuple[str, ...], tuple[np.dtype[Any], ...]
        ] = {}
        self._red_code = red_code
        self._use_common_type = use_common_type

    @staticmethod
    def _find_common_type(
        arrs: Sequence[ndarray], orig_args: Sequence[Any]
    ) -> np.dtype[Any]:
        # FIXME: The following is a miserable attempt to implement type
        # coercion rules that try to match NumPy's rules for a subset of cases;
        # for the others, cuNumeric computes a type different from what
        # NumPy produces for the same operands. Type coercion rules shouldn't
        # be this difficult to imitate...

        all_scalars = all(arr.ndim == 0 for arr in arrs)
        all_arrays = all(arr.ndim > 0 for arr in arrs)
        kinds = set(arr.dtype.kind for arr in arrs)
        lossy_conversion = ("i" in kinds or "u" in kinds) and (
            "f" in kinds or "c" in kinds
        )
        use_min_scalar = not (all_scalars or all_arrays or lossy_conversion)

        scalar_types = []
        array_types = []
        for arr, orig_arg in zip(arrs, orig_args):
            if arr.ndim == 0:
                scalar_types.append(
                    np.dtype(np.min_scalar_type(orig_arg))
                    if use_min_scalar
                    else arr.dtype
                )
            else:
                array_types.append(arr.dtype)

        return np.find_common_type(array_types, scalar_types)

    def _resolve_dtype(
        self,
        arrs: Sequence[ndarray],
        orig_args: Sequence[np.dtype[Any]],
        casting: CastingKind,
        precision_fixed: bool,
    ) -> tuple[Sequence[ndarray], np.dtype[Any]]:
        to_dtypes: tuple[np.dtype[Any], ...]
        key: tuple[str, ...]
        if self._use_common_type:
            common_dtype = self._find_common_type(arrs, orig_args)
            to_dtypes = (common_dtype, common_dtype)
            key = (common_dtype.char, common_dtype.char)
        else:
            to_dtypes = tuple(arr.dtype for arr in arrs)
            key = tuple(arr.dtype.char for arr in arrs)

        if key in self._types:
            arrs = [
                arr.astype(to_dtype) for arr, to_dtype in zip(arrs, to_dtypes)
            ]
            return arrs, np.dtype(self._types[key])

        if not precision_fixed:
            if key in self._resolution_cache:
                to_dtypes = self._resolution_cache[key]
                arrs = [
                    arr.astype(to_dtype)
                    for arr, to_dtype in zip(arrs, to_dtypes)
                ]
                return arrs, np.dtype(self._types[to_dtypes])

        chosen = None
        if not precision_fixed:
            for in_dtypes in self._types.keys():
                if all(
                    np.can_cast(arr.dtype, to_dtype)
                    for arr, to_dtype in zip(arrs, in_dtypes)
                ):
                    chosen = in_dtypes
                    break

            # If there's no safe match and the operands have different types,
            # try to find a match based on the leading operand
            if chosen is None and not self._use_common_type:
                for in_dtypes in self._types.keys():
                    if np.can_cast(arrs[0].dtype, in_dtypes[0]) and all(
                        np.can_cast(arr, to_dtype, casting=casting)
                        for arr, to_dtype in zip(arrs[1:], in_dtypes[1:])
                    ):
                        chosen = in_dtypes
                        break

        if chosen is None:
            raise TypeError(
                f"No matching signature of ufunc {self._name} is found "
                "for the given casting"
            )

        self._resolution_cache[key] = chosen
        arrs = [arr.astype(to_dtype) for arr, to_dtype in zip(arrs, chosen)]

        return arrs, np.dtype(self._types[chosen])

    def __call__(
        self,
        *args: Any,
        out: Union[ndarray, None] = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: Union[np.dtype[Any], None] = None,
        **kwargs: Any,
    ) -> ndarray:
        arrs, (out,), out_shape, where = self._prepare_operands(
            *args, out=out, where=where
        )
        orig_args = args[: self.nin]

        # If no dtype is given to prescribe the accuracy, we use the dtype
        # of the input
        precision_fixed = False
        if dtype is not None:
            # If a dtype is given, that determines the precision
            # of the computation.
            precision_fixed = True
            arrs = [
                self._maybe_cast_input(arr, dtype, casting) for arr in arrs
            ]

        # Resolve the dtype to use for the computation and cast the input
        # if necessary. If the dtype is already fixed by the caller,
        # the dtype must be one of the dtypes supported by this operation.
        arrs, res_dtype = self._resolve_dtype(
            arrs, orig_args, casting, precision_fixed
        )

        x1, x2 = arrs
        result = self._maybe_create_result(
            out, out_shape, res_dtype, casting, (x1, x2, where)
        )
        result._thunk.binary_op(self._op_code, x1._thunk, x2._thunk, where, ())

        return self._maybe_cast_output(out, result)

    def reduce(
        self,
        array: ndarray,
        axis: Union[int, tuple[int, ...], None] = 0,
        dtype: Union[np.dtype[Any], None] = None,
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
        initial: Union[Any, None] = None,
        where: bool = True,
    ) -> ndarray:
        """
        reduce(array, axis=0, dtype=None, out=None, keepdims=False, initial=<no
        value>, where=True)

        Reduces `array`'s dimension by one, by applying ufunc along one axis.

        For example, add.reduce() is equivalent to sum().

        Parameters
        ----------
        array : array_like
            The array to act on.
        axis : None or int or tuple of ints, optional
            Axis or axes along which a reduction is performed.  The default
            (`axis` = 0) is perform a reduction over the first dimension of the
            input array. `axis` may be negative, in which case it counts from
            the last to the first axis.
        dtype : data-type code, optional
            The type used to represent the intermediate results. Defaults to
            the data-type of the output array if this is provided, or the
            data-type
            of the input array if no output array is provided.
        out : ndarray, None, or tuple of ndarray and None, optional
            A location into which the result is stored. If not provided or
            None, a freshly-allocated array is returned. For consistency with
            ``ufunc.__call__``, if given as a keyword, this may be wrapped in a
            1-element tuple.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the original `array`.
        initial : scalar, optional
            The value with which to start the reduction.  If the ufunc has no
            identity or the dtype is object, this defaults to None - otherwise
            it defaults to ufunc.identity.  If ``None`` is given, the first
            element of the reduction is used, and an error is thrown if the
            reduction is empty.
        where : array_like of bool, optional
            A boolean array which is broadcasted to match the dimensions of
            `array`, and selects elements to include in the reduction. Note
            that for ufuncs like ``minimum`` that do not have an identity
            defined, one has to pass in also ``initial``.

        Returns
        -------
        r : ndarray
            The reduced array. If `out` was supplied, `r` is a reference to it.

        See Also
        --------
        numpy.ufunc.reduce
        """
        array = convert_to_cunumeric_ndarray(array)

        if self._red_code is None:
            raise NotImplementedError(
                f"reduction for {self} is not yet implemented"
            )
        if not isinstance(where, bool) or not where:
            raise NotImplementedError(
                "the 'where' keyword is not yet supported"
            )

        # NumPy seems to be using None as the default axis value for scalars
        if array.ndim == 0 and axis == 0:
            axis = None

        # TODO: Unary reductions still need to be refactored
        return array._perform_unary_reduction(
            self._red_code,
            array,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )


def _parse_unary_ufunc_type(ty: str) -> tuple[str, str]:
    if len(ty) == 1:
        return (ty, ty)
    return (ty[0], ty[1:])


def create_unary_ufunc(
    summary: str,
    name: str,
    op_code: UnaryOpCode,
    types: Sequence[str],
    overrides: dict[str, UnaryOpCode] = {},
) -> unary_ufunc:
    doc = _UNARY_DOCSTRING_TEMPLATE.format(summary, name)
    types_dict = dict(_parse_unary_ufunc_type(ty) for ty in types)
    return unary_ufunc(name, doc, op_code, types_dict, overrides)


def create_multiout_unary_ufunc(
    summary: str, name: str, op_code: UnaryOpCode, types: Sequence[str]
) -> multiout_unary_ufunc:
    doc = _MULTIOUT_UNARY_DOCSTRING_TEMPLATE.format(summary, name)
    types_dict = dict(_parse_unary_ufunc_type(ty) for ty in types)
    return multiout_unary_ufunc(name, doc, op_code, types_dict)


def _parse_binary_ufunc_type(ty: str) -> tuple[tuple[str, str], str]:
    if len(ty) == 1:
        return ((ty, ty), ty)
    if len(ty) == 3:
        return ((ty[0], ty[1]), ty[2])
    raise NotImplementedError(
        "Binary ufunc must have two inputs and one output"
    )


def create_binary_ufunc(
    summary: str,
    name: str,
    op_code: BinaryOpCode,
    types: Sequence[str],
    red_code: Union[UnaryRedCode, None] = None,
    use_common_type: bool = True,
) -> binary_ufunc:
    doc = _BINARY_DOCSTRING_TEMPLATE.format(summary, name)
    types_dict = dict(_parse_binary_ufunc_type(ty) for ty in types)
    return binary_ufunc(
        name,
        doc,
        op_code,
        types_dict,
        red_code=red_code,
        use_common_type=use_common_type,
    )
