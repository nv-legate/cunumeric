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

tanh = create_unary_ufunc(
    "Compute hyperbolic tangent element-wise.",
    "tanh",
    UnaryOpCode.TANH,
    ["e", "f", "d", "F", "D"],
)
