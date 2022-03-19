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

negative = create_unary_ufunc(
    "Numerical negative, element-wise.",
    "negative",
    UnaryOpCode.NEGATIVE,
    [
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
        "e",
        "f",
        "d",
        "F",
        "D",
    ],
)

absolute = create_unary_ufunc(
    "Calculate the absolute value element-wise.",
    "absolute",
    UnaryOpCode.ABSOLUTE,
    [
        "?",
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
        "e",
        "f",
        "d",
        "Ff",
        "Dd",
    ],
)

abs = absolute

fabs = create_unary_ufunc(
    "Compute the absolute values element-wise.",
    "fabs",
    UnaryOpCode.ABSOLUTE,
    ["e", "f", "d"],
)

rint = create_unary_ufunc(
    "Round elements of the array to the nearest integer.",
    "rint",
    UnaryOpCode.RINT,
    ["e", "f", "d"],
)

sign = create_unary_ufunc(
    "Returns an element-wise indication of the sign of a number.",
    "sign",
    UnaryOpCode.SIGN,
    [
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
        "e",
        "f",
        "d",
        "F",
        "D",
    ],
)

conjugate = create_unary_ufunc(
    "Return the complex conjugate, element-wise.",
    "conjugate",
    UnaryOpCode.CONJ,
    [
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
        "e",
        "f",
        "d",
        "g",
        "F",
        "D",
    ],
)

conj = conjugate

exp = create_unary_ufunc(
    "Calculate the exponential of all elements in the input array.",
    "exp",
    UnaryOpCode.EXP,
    ["e", "f", "d", "F", "D"],
)

exp2 = create_unary_ufunc(
    "Calculate `2**p` for all `p` in the input array.",
    "exp2",
    UnaryOpCode.EXP2,
    ["e", "f", "d", "F", "D"],
)

log = create_unary_ufunc(
    "Natural logarithm, element-wise.",
    "log",
    UnaryOpCode.LOG,
    ["e", "f", "d", "F", "D"],
)

log10 = create_unary_ufunc(
    "Return the base 10 logarithm of the input array, element-wise.",
    "log10",
    UnaryOpCode.LOG10,
    ["e", "f", "d", "F", "D"],
)

sqrt = create_unary_ufunc(
    "Return the non-negative square-root of an array, element-wise.",
    "sqrt",
    UnaryOpCode.SQRT,
    ["e", "f", "d", "F", "D"],
)
