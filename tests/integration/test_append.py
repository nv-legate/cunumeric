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

import numpy as np
import pytest

import cunumeric as num


def run_test(arr, values, test_args):
    for axis in test_args:
        b = np.append(arr, values, axis)
        c = num.append(arr, values, axis)
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
        print_msg = (
            f"np.append(array({arr.shape}), array({values.shape}), {axis})"
        )
        assert is_equal, (
            f"Failed, {print_msg}\n"
            f"numpy result: {err_arr[0]}, {b.shape}\n"
            f"cunumeric_result: {err_arr[1]}, {c.shape}\n"
            f"cunumeric and numpy shows"
            f" different result\n"
        )
        print(
            f"Passed, {print_msg}, np: ({b.shape}, {b.dtype})"
            f", cunumeric: ({c.shape}, {c.dtype}"
        )


def test():
    dim = 10
    print("test append")
    # test append w/ 1D, 2D and 3D arrays
    input_arr = [
        (0,),
        (0, 10),
        (1,),
        (1, 1),
        (1, 1, 1),
        (1, dim),
        (dim, dim),
        (dim, dim, dim),
    ]
    for input_size in input_arr:
        a = np.random.randint(low=0, high=100, size=(input_size))

        test_args = list(range(a.ndim))
        test_args.append(None)
        # test the exception for 1D array on append
        run_test(a, a, test_args)
        if a.ndim > 1:
            # 1D array
            b = np.random.randint(low=0, high=100, size=(dim,))
            run_test(a, b, [None])


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
