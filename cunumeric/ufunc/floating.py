# Copyright 2022 NVIDIA Corporation
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

from cunumeric.config import UnaryOpCode

from .ufunc import (
    create_unary_ufunc,
    float_and_complex,
    float_dtypes,
    predicate_types_of,
)

isfinite = create_unary_ufunc(
    "Test element-wise for finiteness (not infinity and not Not a Number).",
    "isfinite",
    UnaryOpCode.ISFINITE,
    predicate_types_of(float_and_complex),
)


isinf = create_unary_ufunc(
    "Test element-wise for positive or negative infinity.",
    "isinf",
    UnaryOpCode.ISINF,
    predicate_types_of(float_and_complex),
)


isnan = create_unary_ufunc(
    "Test element-wise for NaN and return result as a boolean array.",
    "isnan",
    UnaryOpCode.ISNAN,
    predicate_types_of(float_and_complex),
)


fabs = create_unary_ufunc(
    "Compute the absolute values element-wise.",
    "fabs",
    UnaryOpCode.ABSOLUTE,
    float_dtypes,
)


signbit = create_unary_ufunc(
    "Returns element-wise True where signbit is set (less than zero).",
    "signbit",
    UnaryOpCode.SIGNBIT,
    predicate_types_of(float_dtypes),
)


floor = create_unary_ufunc(
    "Return the floor of the input, element-wise.",
    "floor",
    UnaryOpCode.FLOOR,
    float_dtypes,
)

ceil = create_unary_ufunc(
    "Return the ceiling of the input, element-wise.",
    "ceil",
    UnaryOpCode.CEIL,
    float_dtypes,
)

trunc = create_unary_ufunc(
    "Return the truncated value of the input, element-wise.",
    "trunc",
    UnaryOpCode.TRUNC,
    float_and_complex,
)
