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

from .ufunc import create_unary_ufunc

isfinite = create_unary_ufunc(
    "Test element-wise for finiteness (not infinity and not Not a Number).",
    "isfinite",
    UnaryOpCode.ISFINITE,
    ["e?", "f?", "d?", "F?", "D?"],
)


isinf = create_unary_ufunc(
    "Test element-wise for positive or negative infinity.",
    "isinf",
    UnaryOpCode.ISINF,
    ["e?", "f?", "d?", "F?", "D?"],
)


isnan = create_unary_ufunc(
    "Test element-wise for NaN and return result as a boolean array.",
    "isnan",
    UnaryOpCode.ISNAN,
    ["e?", "f?", "d?", "F?", "D?"],
)


fabs = create_unary_ufunc(
    "Compute the absolute values element-wise.",
    "fabs",
    UnaryOpCode.ABSOLUTE,
    ["e", "f", "d"],
)


signbit = create_unary_ufunc(
    "Returns element-wise True where signbit is set (less than zero).",
    "signbit",
    UnaryOpCode.SIGNBIT,
    ["e?", "f?", "d?"],
)


floor = create_unary_ufunc(
    "Return the floor of the input, element-wise.",
    "floor",
    UnaryOpCode.FLOOR,
    ["e", "f", "d"],
)

ceil = create_unary_ufunc(
    "Return the ceiling of the input, element-wise.",
    "ceil",
    UnaryOpCode.CEIL,
    ["e", "f", "d"],
)

trunc = create_unary_ufunc(
    "Return the truncated value of the input, element-wise.",
    "trunc",
    UnaryOpCode.TRUNC,
    ["e", "f", "d", "F", "D"],
)
