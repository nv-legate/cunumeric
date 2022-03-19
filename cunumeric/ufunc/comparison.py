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

logical_not = create_unary_ufunc(
    "Compute bit-wise inversion, or bit-wise NOT, element-wise.",
    "invert",
    UnaryOpCode.LOGICAL_NOT,
    [
        "??",
        "b?",
        "B?",
        "h?",
        "H?",
        "i?",
        "I?",
        "l?",
        "L?",
        "q?",
        "Q?",
        "e?",
        "f?",
        "d?",
    ],
    overrides={"?": UnaryOpCode.LOGICAL_NOT},
)
