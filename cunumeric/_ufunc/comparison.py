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
from __future__ import annotations

from cunumeric.config import BinaryOpCode, UnaryOpCode, UnaryRedCode

from .ufunc import (
    all_dtypes,
    create_binary_ufunc,
    create_unary_ufunc,
    float_dtypes,
    integer_dtypes,
    predicate_types_of,
    relation_types_of,
)

greater = create_binary_ufunc(
    "Return the truth value of (x1 > x2) element-wise.",
    "greater",
    BinaryOpCode.GREATER,
    relation_types_of(all_dtypes),
)

greater_equal = create_binary_ufunc(
    "Return the truth value of (x1 >= x2) element-wise.",
    "greater_equal",
    BinaryOpCode.GREATER_EQUAL,
    relation_types_of(all_dtypes),
)

less = create_binary_ufunc(
    "Return the truth value of (x1 < x2) element-wise.",
    "less",
    BinaryOpCode.LESS,
    relation_types_of(all_dtypes),
)

less_equal = create_binary_ufunc(
    "Return the truth value of (x1 =< x2) element-wise.",
    "less",
    BinaryOpCode.LESS_EQUAL,
    relation_types_of(all_dtypes),
)

not_equal = create_binary_ufunc(
    "Return (x1 != x2) element-wise.",
    "not_equal",
    BinaryOpCode.NOT_EQUAL,
    relation_types_of(all_dtypes),
)

equal = create_binary_ufunc(
    "Return (x1 == x2) element-wise.",
    "equal",
    BinaryOpCode.EQUAL,
    relation_types_of(all_dtypes),
)

logical_and = create_binary_ufunc(
    "Compute the truth value of x1 AND x2 element-wise.",
    "logical_and",
    BinaryOpCode.LOGICAL_AND,
    relation_types_of(all_dtypes),
    red_code=UnaryRedCode.ALL,
)

logical_or = create_binary_ufunc(
    "Compute the truth value of x1 OR x2 element-wise.",
    "logical_or",
    BinaryOpCode.LOGICAL_OR,
    relation_types_of(all_dtypes),
    red_code=UnaryRedCode.ANY,
)

logical_xor = create_binary_ufunc(
    "Compute the truth value of x1 XOR x2, element-wise.",
    "logical_xor",
    BinaryOpCode.LOGICAL_XOR,
    relation_types_of(all_dtypes),
)

logical_not = create_unary_ufunc(
    "Compute bit-wise inversion, or bit-wise NOT, element-wise.",
    "invert",
    UnaryOpCode.LOGICAL_NOT,
    (
        ["??"]
        + predicate_types_of(integer_dtypes)
        + predicate_types_of(float_dtypes)
    ),
    overrides={"?": UnaryOpCode.LOGICAL_NOT},
)

maximum = create_binary_ufunc(
    "Element-wise maximum of array elements.",
    "maximum",
    BinaryOpCode.MAXIMUM,
    all_dtypes,
    red_code=UnaryRedCode.MAX,
)

fmax = maximum

minimum = create_binary_ufunc(
    "Element-wise minimum of array elements.",
    "minimum",
    BinaryOpCode.MINIMUM,
    all_dtypes,
    red_code=UnaryRedCode.MIN,
)

fmin = minimum
