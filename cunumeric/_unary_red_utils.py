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

from typing import Any, Union

import numpy as np

from .array import ndarray
from .config import UnaryRedCode
from .runtime import runtime

_EQUIVALENT_NON_NAN_UNARY_RED_OPS: dict[UnaryRedCode, UnaryRedCode] = {
    UnaryRedCode.NANARGMAX: UnaryRedCode.ARGMAX,
    UnaryRedCode.NANARGMIN: UnaryRedCode.ARGMIN,
    UnaryRedCode.NANMAX: UnaryRedCode.MAX,
    UnaryRedCode.NANMIN: UnaryRedCode.MIN,
}

_UNARY_RED_OPS_EXCEPTIONS_HANDLED = (
    UnaryRedCode.NANMIN,
    UnaryRedCode.NANMAX,
    UnaryRedCode.NANARGMIN,
    UnaryRedCode.NANARGMAX,
)


def get_non_nan_unary_red_code(
    kind: str, unary_red_code: UnaryRedCode
) -> UnaryRedCode:
    """
    Return the equivalent non-nan reduction op codes if the datatype
    of the array isn't floating-point. Raise an error if the datatype
    is disallowed, which is currently complex64 and complex128.
    """

    assert unary_red_code in _EQUIVALENT_NON_NAN_UNARY_RED_OPS

    # complex datatype is not supported
    if kind == "c":
        raise NotImplementedError(
            "operation is not supported for complex64 and complex128 types"
        )

    # use NaN API if the datatype is floating point type otherwise
    # use equivalent non-NaN API
    if kind == "f":
        return unary_red_code

    return _EQUIVALENT_NON_NAN_UNARY_RED_OPS[unary_red_code]


def handle_nan_unary_red_exceptions(
    out: ndarray, op: UnaryRedCode, identity: Union[int, np.floating[Any]]
) -> None:
    """Raise ValueError if NaN is found in a slice or in the entire
    array for nanargmin and nanargmax. In case of nanmin and nanmax,
    issue a RuntimeWarning.
    """

    if op not in _UNARY_RED_OPS_EXCEPTIONS_HANDLED:
        return

    if identity in out:
        if op in (UnaryRedCode.NANMIN, UnaryRedCode.NANMAX):
            runtime.warn(
                "Array/Slice contains only NaNs", category=RuntimeWarning
            )
        else:
            raise ValueError("Array/Slice contains only NaNs")
