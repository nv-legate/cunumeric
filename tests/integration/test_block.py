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


def run_test(arr, input_sizes, depth):
    #    print (arr)
    b = np.block(arr)
    c = num.block(arr)
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
    for i in range(depth):
        input_sizes = [input_sizes]
    print_msg = f"np.block([{input_sizes}, {input_sizes}])"
    assert is_equal, (
        f"Failed, {print_msg}\n"
        f"numpy result: {err_arr[0]}\n"
        f"cunumeric_result: {err_arr[1]}\n"
        f"cunumeric and numpy shows"
        f" different result\n"
    )
    print(
        f"Passed, {print_msg}, np: ({b.shape}, {b.dtype})"
        f", cunumeric: ({c.shape}, {c.dtype}"
    )


def test():
    dim = 10
    print("test np.block")
    # test append w/ 1D, 2D and 3D arrays
    input_options = [
        ((0,), (0,)),  # empty arrays
        ((1,), (1,)),  # singlton arrays
        ((dim, 1), (dim, dim)),  # 1D and 2D arrays
        ((dim, 1), (dim, 1), (dim, dim)),  # 3 arrays in the inner-most list
    ]
    for input_sizes in input_options:
        for depth in range(2):  # test depth from 1 to 3
            a = list(
                np.arange(np.prod(input_size)).reshape(input_size)
                for input_size in input_sizes
            )
            b = list(
                np.arange(np.prod(input_size)).reshape(input_size)
                for input_size in input_sizes
            )
            for nlist in range(depth):
                a = [a]
                b = [b]
            run_test([a, b], input_sizes, depth)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
