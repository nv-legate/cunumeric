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
