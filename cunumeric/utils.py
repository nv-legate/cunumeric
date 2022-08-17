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

import traceback
from functools import reduce
from string import ascii_lowercase, ascii_uppercase
from types import FrameType
from typing import Any, List, Sequence, Tuple, Union, cast

import numpy as np

from .types import NdShape, NdShapeLike

_SUPPORTED_DTYPES = [
    np.float16,
    np.float32,
    np.float64,
    float,
    np.int16,
    np.int32,
    np.int64,
    int,
    np.uint16,
    np.uint32,
    np.uint64,
    np.bool_,
    bool,
]


def _broadcast_shapes(*args: Sequence[NdShapeLike]) -> tuple[int]:
    # Call _broadcast_shapes' for now.
    # We will have a new implementation later
    return np.broadcast_shapes(*args)


def is_advanced_indexing(key: Any) -> bool:
    if key is Ellipsis or key is None:  # np.newdim case
        return False
    if np.isscalar(key):
        return False
    if isinstance(key, slice):
        return False
    if isinstance(key, tuple):
        return any(is_advanced_indexing(k) for k in key)
    # Any other kind of thing leads to advanced indexing
    return True


def find_last_user_stacklevel() -> int:
    stacklevel = 1
    for (frame, _) in traceback.walk_stack(None):
        if not frame.f_globals["__name__"].startswith("cunumeric"):
            break
        stacklevel += 1
    return stacklevel


def get_line_number_from_frame(frame: FrameType) -> str:
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


def find_last_user_frames(top_only: bool = True) -> str:
    for (last, _) in traceback.walk_stack(None):
        if "__name__" not in last.f_globals:
            continue
        if not last.f_globals["__name__"].startswith("cunumeric"):
            break

    if top_only:
        return get_line_number_from_frame(last)

    frames: list[FrameType] = []
    curr: Union[FrameType, None] = last
    while curr is not None:
        if "legion_top.py" in curr.f_code.co_filename:
            break
        frames.append(curr)
        curr = curr.f_back
    return "|".join(get_line_number_from_frame(f) for f in frames)


def is_supported_dtype(dtype: Any) -> bool:
    if not isinstance(dtype, np.dtype):
        raise TypeError("expected a NumPy dtype")
    return dtype.type in _SUPPORTED_DTYPES


def calculate_volume(shape: NdShape) -> int:
    if len(shape) == 0:
        return 0
    return reduce(lambda x, y: x * y, shape)


def get_arg_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    return np.dtype(
        [("arg", np.int64), ("arg_value", dtype)],
        align=True,
    )


def get_arg_value_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    dt = dtype.fields["arg_value"][0].type  # type: ignore [index]
    return cast(Any, dt)


Modes = Tuple[List[str], List[str], List[str]]


def dot_modes(a_ndim: int, b_ndim: int) -> Modes:
    a_modes = list(ascii_lowercase[:a_ndim])
    b_modes = list(ascii_uppercase[:b_ndim])
    if a_ndim == 0 or b_ndim == 0:
        out_modes = a_modes + b_modes
    elif b_ndim == 1:
        b_modes[-1] = a_modes[-1]
        out_modes = a_modes[:-1]
    else:
        b_modes[-2] = a_modes[-1]
        out_modes = a_modes[:-1] + b_modes[:-2] + [b_modes[-1]]
    return (a_modes, b_modes, out_modes)


def inner_modes(a_ndim: int, b_ndim: int) -> Modes:
    a_modes = list(ascii_lowercase[:a_ndim])
    b_modes = list(ascii_uppercase[:b_ndim])
    if a_ndim == 0 or b_ndim == 0:
        out_modes = a_modes + b_modes
    else:
        b_modes[-1] = a_modes[-1]
        out_modes = a_modes[:-1] + b_modes[:-1]
    return (a_modes, b_modes, out_modes)


def matmul_modes(a_ndim: int, b_ndim: int) -> Modes:
    if a_ndim == 0 or b_ndim == 0:
        raise ValueError("Scalars not allowed in matmul")
    a_modes = list(ascii_lowercase[-a_ndim:])
    b_modes = list(ascii_lowercase[-b_ndim:])
    if b_ndim >= 2:
        a_modes[-1] = "A"
        b_modes[-2] = "A"
    if b_ndim == 1:
        out_modes = a_modes[:-1]
    elif a_ndim == 1:
        out_modes = b_modes[:-2] + [b_modes[-1]]
    else:
        out_modes = (
            list(ascii_lowercase[-max(a_ndim, b_ndim) : -2])
            + [a_modes[-2]]
            + [b_modes[-1]]
        )
    return (a_modes, b_modes, out_modes)


Axes = Sequence[int]
AxesPair = Tuple[Axes, Axes]
AxesPairLikeTuple = Union[
    Tuple[int, int],
    Tuple[int, Axes],
    Tuple[Axes, int],
    Tuple[Axes, Axes],
]
AxesPairLike = Union[int, AxesPairLikeTuple]


def tensordot_modes(a_ndim: int, b_ndim: int, axes: AxesPairLike) -> Modes:
    def convert_int_axes(axes: int) -> AxesPair:
        return list(range(a_ndim - axes, a_ndim)), list(range(axes))

    def convert_seq_axes(axes: AxesPairLikeTuple) -> AxesPair:
        a_axes, b_axes = axes
        return (
            [a_axes] if isinstance(a_axes, int) else list(a_axes),
            [b_axes] if isinstance(b_axes, int) else list(b_axes),
        )

    def convert_axes(axes: AxesPairLike) -> AxesPair:
        if isinstance(axes, int):
            a_axes, b_axes = convert_int_axes(axes)
        else:
            a_axes, b_axes = convert_seq_axes(axes)

        return (
            [ax + a_ndim if ax < 0 else ax for ax in a_axes],
            [ax + b_ndim if ax < 0 else ax for ax in b_axes],
        )

    def check_axes(a_axes: Axes, b_axes: Axes) -> None:
        if (
            len(a_axes) != len(b_axes)
            or len(a_axes) > a_ndim
            or len(b_axes) > b_ndim
            or len(a_axes) != len(set(a_axes))
            or len(b_axes) != len(set(b_axes))
            or any(ax < 0 for ax in a_axes)
            or any(ax < 0 for ax in b_axes)
            or any(ax >= a_ndim for ax in a_axes)
            or any(ax >= b_ndim for ax in b_axes)
        ):
            raise ValueError("Invalid axes argument")

    a_axes, b_axes = convert_axes(axes)

    check_axes(a_axes, b_axes)

    a_modes = list(ascii_lowercase[:a_ndim])
    b_modes = list(ascii_uppercase[:b_ndim])
    for (a_i, b_i) in zip(a_axes, b_axes):
        b_modes[b_i] = a_modes[a_i]
    a_out = [a_modes[a_i] for a_i in sorted(set(range(a_ndim)) - set(a_axes))]
    b_out = [b_modes[b_i] for b_i in sorted(set(range(b_ndim)) - set(b_axes))]

    return (a_modes, b_modes, a_out + b_out)
