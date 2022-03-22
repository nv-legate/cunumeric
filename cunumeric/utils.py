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

import traceback
from functools import reduce
from string import ascii_lowercase, ascii_uppercase

import numpy as np


def find_last_user_stacklevel():
    stacklevel = 1
    for (frame, _) in traceback.walk_stack(None):
        if not frame.f_globals["__name__"].startswith("cunumeric"):
            break
        stacklevel += 1
    return stacklevel


def get_line_number_from_frame(frame):
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


def find_last_user_frames(top_only=True):
    last = None
    for (frame, _) in traceback.walk_stack(None):
        last = frame
        if "__name__" not in frame.f_globals:
            continue
        if not frame.f_globals["__name__"].startswith("cunumeric"):
            break

    if top_only:
        return get_line_number_from_frame(last)

    frames = []
    curr = last
    while curr is not None:
        if "legion_top.py" in curr.f_code.co_filename:
            break
        frames.append(curr)
        curr = curr.f_back
    return "|".join(get_line_number_from_frame(f) for f in frames)


# These are the dtypes that we currently support for cuNumeric
def is_supported_dtype(dtype):
    assert isinstance(dtype, np.dtype)
    base_type = dtype.type
    if (
        base_type == np.float16
        or base_type == np.float32
        or base_type == np.float64
        or base_type == float
    ):
        return True
    if (
        base_type == np.int16
        or base_type == np.int32
        or base_type == np.int64
        or base_type == int
    ):
        return True
    if (
        base_type == np.uint16
        or base_type == np.uint32
        or base_type == np.uint64
    ):  # noqa E501
        return True
    if base_type == np.bool_ or base_type == bool:
        return True
    return False


def calculate_volume(shape):
    if shape == ():
        return 0
    return reduce(lambda x, y: x * y, shape)


def get_arg_dtype(dtype):
    return np.dtype(
        [("arg", np.int64), ("arg_value", dtype)],
        align=True,
    )


def get_arg_value_dtype(dtype):
    return dtype.fields["arg_value"][0].type


def dot_modes(a_ndim, b_ndim):
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


def inner_modes(a_ndim, b_ndim):
    a_modes = list(ascii_lowercase[:a_ndim])
    b_modes = list(ascii_uppercase[:b_ndim])
    if a_ndim == 0 or b_ndim == 0:
        out_modes = a_modes + b_modes
    else:
        b_modes[-1] = a_modes[-1]
        out_modes = a_modes[:-1] + b_modes[:-1]
    return (a_modes, b_modes, out_modes)


def matmul_modes(a_ndim, b_ndim):
    assert a_ndim >= 1 and b_ndim >= 1
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


def tensordot_modes(a_ndim, b_ndim, axes):
    if isinstance(axes, int):
        if axes > a_ndim or axes > b_ndim:
            raise ValueError("Invalid axes argument")
        axes = (list(range(a_ndim - axes, a_ndim)), list(range(axes)))
    a_axes, b_axes = axes
    if isinstance(a_axes, int):
        a_axes = [a_axes]
    if isinstance(b_axes, int):
        b_axes = [b_axes]
    a_axes = [ax + a_ndim if ax < 0 else ax for ax in a_axes]
    b_axes = [ax + b_ndim if ax < 0 else ax for ax in b_axes]
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

    a_modes = list(ascii_lowercase[:a_ndim])
    b_modes = list(ascii_uppercase[:b_ndim])
    for (a_i, b_i) in zip(a_axes, b_axes):
        b_modes[b_i] = a_modes[a_i]
    out_modes = [
        a_modes[a_i] for a_i in sorted(set(range(a_ndim)) - set(a_axes))
    ] + [b_modes[b_i] for b_i in sorted(set(range(b_ndim)) - set(b_axes))]
    return (a_modes, b_modes, out_modes)
