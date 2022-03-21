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

from cunumeric.array import (
    broadcast_shapes,
    convert_to_cunumeric_ndarray,
    ndarray,
)
from numpy import can_cast as np_can_cast, dtype as np_dtype

_DOCSTRING_TEMPLATE = """{}

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


def predicate_types_of(dtypes):
    return [ty + "?" for ty in dtypes]


def relation_types_of(dtypes):
    return [ty * 2 + "?" for ty in dtypes]


class ufunc:
    def _maybe_cast_input(self, arr, to_dtype, casting):
        if arr.dtype == to_dtype:
            return arr

        if not np_can_cast(arr.dtype, to_dtype):
            raise TypeError(
                f"Cannot cast ufunc '{self._name}' input from "
                f"{arr.dtype} to {to_dtype} with casting rule '{casting}'"
            )

        return arr.astype(to_dtype)


class unary_ufunc(ufunc):
    def __init__(self, name, doc, op_code, types, overrides):
        self._name = name
        self._op_code = op_code
        self._types = types
        self._resolution_cache = {}
        self.__doc__ = doc
        self._overrides = overrides

    @property
    def nin(self):
        return 1

    @property
    def nout(self):
        return 1

    @property
    def types(self):
        return [f"{in_ty}->{out_ty}" for in_ty, out_ty in self._types.items()]

    @property
    def ntypes(self):
        return len(self._types)

    def _resolve_dtype(self, arr, casting, precision_fixed):
        if arr.dtype.char in self._types:
            return arr, np_dtype(self._types[arr.dtype.char])

        if arr.dtype in self._resolution_cache:
            to_dtype = self._resolution_cache[arr.dtype]
            arr = arr.astype(to_dtype)
            return arr, np_dtype(self._types[to_dtype.char])

        chosen = None
        if not precision_fixed:
            for in_ty in self._types.keys():
                if np_can_cast(arr.dtype, in_ty):
                    chosen = in_ty
                    break

        if chosen is None:
            raise TypeError(
                f"No matching signature of ufunc {self._name} is found "
                "for the given casting"
            )

        to_dtype = np_dtype(chosen)
        self._resolution_cache[arr.dtype] = to_dtype

        return arr.astype(to_dtype), np_dtype(self._types[to_dtype.char])

    def __call__(
        self,
        x,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        **kwargs,
    ):
        x = convert_to_cunumeric_ndarray(x)

        if out is not None:
            if isinstance(out, tuple):
                if len(out) != 1:
                    raise ValueError(
                        "The 'out' tuple must have exactly one entry "
                        "per ufunc output"
                    )
                out = out[0]

            if not isinstance(out, ndarray):
                raise TypeError("return arrays must be of ArrayType")

            # Check if the broadcasting is possible
            broadcast_shapes(x.shape, out.shape)

        if not isinstance(where, bool) or not where:
            raise NotImplementedError(
                "the 'where' keyword is not yet supported"
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
        x, res_dtype = self._resolve_dtype(x, casting, precision_fixed)

        if out is None:
            result = ndarray(shape=x.shape, dtype=res_dtype, inputs=(x, where))
            out = result
        else:
            if out.dtype != res_dtype:
                if not np_can_cast(res_dtype, out.dtype):
                    raise TypeError(
                        f"Cannot cast ufunc '{self._name}' output from "
                        f"{res_dtype} to {out.dtype} with casting rule "
                        f"'{casting}'"
                    )
                result = ndarray(
                    shape=out.shape, dtype=res_dtype, inputs=(x, where)
                )
            else:
                result = out

        op_code = self._overrides.get(x.dtype.char, self._op_code)
        result._thunk.unary_op(op_code, x._thunk, where, ())

        if out is not result:
            out._thunk.convert(result._thunk, warn=False)

        return out

    def __repr__(self):
        return f"<ufunc {self._name}>"


class binary_ufunc(ufunc):
    def __init__(self, name, doc, op_code, types):
        self._name = name
        self._op_code = op_code
        self._types = types
        self._resolution_cache = {}
        self.__doc__ = doc

    @property
    def nin(self):
        return 2

    @property
    def nout(self):
        return 1

    @property
    def types(self):
        return [
            f"{''.join(in_tys)}->{out_ty}"
            for in_tys, out_ty in self._types.items()
        ]

    @property
    def ntypes(self):
        return len(self._types)

    def _resolve_dtype(self, arrs, casting, precision_fixed):
        common_dtype = ndarray.find_common_type(*arrs)

        key = (common_dtype.char, common_dtype.char)
        if key in self._types:
            arrs = [arr.astype(common_dtype) for arr in arrs]
            return arrs, np_dtype(self._types[key])

        if key in self._resolution_cache:
            to_dtypes = self._resolution_cache[key]
            arrs = [
                arr.astype(to_dtype) for arr, to_dtype in zip(arrs, to_dtypes)
            ]
            return arrs, np_dtype(self._types[to_dtypes])

        chosen = None
        if not precision_fixed:
            for in_dtypes in self._types.keys():
                if all(
                    np_can_cast(arr.dtype, to_dtype)
                    for arr, to_dtype in zip(arrs, in_dtypes)
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

        return arrs, np_dtype(self._types[chosen])

    def __call__(
        self,
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        **kwargs,
    ):
        arrs = [convert_to_cunumeric_ndarray(arr) for arr in (x1, x2)]

        if out is not None:
            if isinstance(out, tuple):
                if len(out) != 1:
                    raise ValueError(
                        "The 'out' tuple must have exactly one entry "
                        "per ufunc output"
                    )
                out = out[0]

            if not isinstance(out, ndarray):
                raise TypeError("return arrays must be of ArrayType")

            # Check if the broadcasting is possible
            out_shape = broadcast_shapes(
                arrs[0].shape, arrs[1].shape, out.shape
            )
        else:
            # Check if the broadcasting is possible
            out_shape = broadcast_shapes(arrs[0].shape, arrs[1].shape)

        if not isinstance(where, bool) or not where:
            raise NotImplementedError(
                "the 'where' keyword is not yet supported"
            )

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
        arrs, res_dtype = self._resolve_dtype(arrs, casting, precision_fixed)

        if out is None:
            result = ndarray(
                shape=out_shape, dtype=res_dtype, inputs=(*arrs, where)
            )
            out = result
        else:
            if out.dtype != res_dtype:
                if not np_can_cast(res_dtype, out.dtype):
                    raise TypeError(
                        f"Cannot cast ufunc '{self._name}' output from "
                        f"{res_dtype} to {out.dtype} with casting rule "
                        f"'{casting}'"
                    )
                result = ndarray(
                    shape=out.shape, dtype=res_dtype, inputs=(*arrs, where)
                )
            else:
                result = out

        x1, x2 = arrs
        print(self._op_code)
        result._thunk.binary_op(self._op_code, x1._thunk, x2._thunk, where, ())

        if out is not result:
            out._thunk.convert(result._thunk, warn=False)

        return out

    def __repr__(self):
        return f"<ufunc {self._name}>"


def _parse_unary_ufunc_type(ty):
    if len(ty) == 1:
        return (ty, ty)
    else:
        if len(ty) > 2:
            raise NotImplementedError("Unary ufunc must have only one output")
        return (ty[0], ty[1])


def create_unary_ufunc(summary, name, op_code, types, overrides={}):
    doc = _DOCSTRING_TEMPLATE.format(summary, name)
    types = dict(_parse_unary_ufunc_type(ty) for ty in types)
    return unary_ufunc(name, doc, op_code, types, overrides)


def _parse_binary_ufunc_type(ty):
    if len(ty) == 1:
        return ((ty, ty), ty)
    else:
        if len(ty) != 3:
            raise NotImplementedError(
                "Binary ufunc must have two inputs and one output"
            )
        elif ty[0] != ty[1]:
            raise NotImplementedError(
                "Operands of binary ufunc must have the same dtype"
            )
        return ((ty[0], ty[1]), ty[2])


def create_binary_ufunc(summary, name, op_code, types):
    doc = _DOCSTRING_TEMPLATE.format(summary, name)
    types = dict(_parse_binary_ufunc_type(ty) for ty in types)
    return binary_ufunc(name, doc, op_code, types)
