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

sin = create_unary_ufunc(
    "Trigonometric sine, element-wise.",
    "sin",
    UnaryOpCode.SIN,
    ["e", "f", "d", "F", "D"],
)

cos = create_unary_ufunc(
    "Cosine element-wise.",
    "cos",
    UnaryOpCode.COS,
    ["e", "f", "d", "F", "D"],
)

tan = create_unary_ufunc(
    "Compute tangent element-wise.",
    "tan",
    UnaryOpCode.TAN,
    ["e", "f", "d", "F", "D"],
)

arcsin = create_unary_ufunc(
    "Inverse sine, element-wise.",
    "arcsin",
    UnaryOpCode.ARCSIN,
    ["e", "f", "d", "F", "D"],
)

arccos = create_unary_ufunc(
    "Trigonometric inverse cosine, element-wise.",
    "arccos",
    UnaryOpCode.ARCCOS,
    ["e", "f", "d", "F", "D"],
)

arctan = create_unary_ufunc(
    "Trigonometric inverse tangent, element-wise.",
    "arctan",
    UnaryOpCode.ARCTAN,
    ["e", "f", "d", "F", "D"],
)

sinh = create_unary_ufunc(
    "Hyperbolic sine, element-wise.",
    "sinh",
    UnaryOpCode.SINH,
    ["e", "f", "d", "F", "D"],
)

cosh = create_unary_ufunc(
    "Hyperbolic cosine, element-wise.",
    "cos",
    UnaryOpCode.COSH,
    ["e", "f", "d", "F", "D"],
)

tanh = create_unary_ufunc(
    "Compute hyperbolic tangent element-wise.",
    "tanh",
    UnaryOpCode.TANH,
    ["e", "f", "d", "F", "D"],
)

arcsinh = create_unary_ufunc(
    "Inverse hyperbolic sine element-wise.",
    "arcsinh",
    UnaryOpCode.ARCSINH,
    ["e", "f", "d", "F", "D"],
)

arccosh = create_unary_ufunc(
    "Inverse hyperbolic cosine, element-wise.",
    "arccosh",
    UnaryOpCode.ARCCOSH,
    ["e", "f", "d", "F", "D"],
)

arctanh = create_unary_ufunc(
    "Inverse hyperbolic tangent element-wise.",
    "arctanh",
    UnaryOpCode.ARCTANH,
    ["e", "f", "d", "F", "D"],
)

deg2rad = create_unary_ufunc(
    "Convert angles from degrees to radians.",
    "deg2rad",
    UnaryOpCode.DEG2RAD,
    ["e", "f", "d"],
)

rad2deg = create_unary_ufunc(
    "Convert angles from radians to degrees.",
    "rad2deg",
    UnaryOpCode.RAD2DEG,
    ["e", "f", "d"],
)

degrees = rad2deg

radians = deg2rad
