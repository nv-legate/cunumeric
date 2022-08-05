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

import numpy as np

import cunumeric as num


def compare_array(b, c):
    """
    Compare two array using zip method.
    """

    is_equal = True
    err_arr = [b, c]

    if len(b) != len(c):
        is_equal = False
        err_arr = [b, c]
    else:
        for each in zip(b, c):
            if not np.array_equal(*each):
                err_arr = each
                is_equal = False
                break
    return is_equal, err_arr


def run_test(fn, args, kwargs, print_msg):
    """
    Run np.func and num.func respectively and compare resutls
    """

    b = getattr(num, fn)(*args, **kwargs)
    c = getattr(np, fn)(*args, **kwargs)

    is_equal, err_arr = compare_array(b, c)

    assert is_equal, (
        f"Failed, {print_msg}\n"
        f"numpy result: {err_arr[0]}, {b.shape}\n"
        f"cunumeric_result: {err_arr[1]}, {c.shape}\n"
        f"cunumeric and numpy shows"
        f" different result\n"
    )
    print(
        f"Passed, {print_msg}, np: ({b.shape}, {b.dtype})"
        f", cunumeric: ({c.shape}, {c.dtype})"
    )
