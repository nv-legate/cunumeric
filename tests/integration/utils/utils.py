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


def compare_array(a, b):
    """
    Compare two array using zip method.
    """

    if len(a) != len(b):
        return False, [a, b]
    else:
        for each in zip(a, b):
            if not np.array_equal(*each):
                return False, each
    return True, None


def check_array_method(fn, args, kwargs, print_msg):
    """
    Run np.func and num.func respectively and compare results
    """

    a = getattr(num, fn)(*args, **kwargs)
    b = getattr(np, fn)(*args, **kwargs)

    is_equal, err_arr = compare_array(a, b)

    assert is_equal, (
        f"Failed, {print_msg}\n"
        f"numpy result: {err_arr[0]}, {a.shape}\n"
        f"cunumeric_result: {err_arr[1]}, {b.shape}\n"
        f"cunumeric and numpy shows"
        f" different result\n"
    )

    if isinstance(a, list):
        print(f"Passed, {print_msg}")
    else:
        print(
            f"Passed, {print_msg}, np: ({a.shape}, {a.dtype})"
            f", cunumeric: ({b.shape}, {b.dtype})"
        )
