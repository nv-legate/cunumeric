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

import itertools

import numpy as np

import cunumeric as num


def run_test(arr, routine, input_size):
    input_arr = [[arr]]
    if routine == "concatenate" or routine == "stack":
        # 'axis' options
        input_arr.append([axis for axis in range(arr[0].ndim)])
        # test axis == 'None' for concatenate
        if routine == "concatenate":
            input_arr[-1].append(None)
        # 'out' argument
        input_arr.append([None])
    test_args = itertools.product(*input_arr)

    for args in test_args:
        b = getattr(np, routine)(*args)
        c = getattr(num, routine)(*args)
        is_equal = True
        err_arr = [b, c]

        if len(b) != len(c):
            is_equal = False
            err_arr = [b, c]
        else:
            for each in zip(b, c):
                sub_set = list(each)
                if not np.array_equal(sub_set[0], sub_set[1]):
                    err_arr = sub_set
                    is_equal = False
                    break
        shape_list = list(inp.shape for inp in arr)
        print_msg = f"np.{routine}(array({shape_list})" f", {args[1:]})"
        assert is_equal, (
            f"Failed, {print_msg}\n"
            f"numpy result: {err_arr[0]}, {b.shape}\n"
            f"cunumeric_result: {err_arr[1]}, {c.shape}\n"
            f"cunumeric and numpy shows"
            f" different result\n"
            f"array({arr}),"
            f"routine: {routine},"
            f"args: {args[1:]}"
        )
        print(
            f"Passed, {print_msg}, np: ({b.shape}, {b.dtype})"
            f", cunumeric: ({c.shape}, {c.dtype}"
        )


def test(dim):
    print("test np.concatenate / *stack")
    test_routine = [
        "concatenate",
        "stack",
        "vstack",
        "hstack",
        "dstack",
        "column_stack",
    ]
    # test np.concatenate & *stack w/ 1D, 2D and 3D arrays
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
    for routine, input_size in itertools.product(test_routine, input_arr):
        a = [
            np.random.randint(low=0, high=100, size=(input_size))
            for num_arr in range(3)
        ]
        # test the exception for 1D array on vstack and dstack
        if routine == "vstack" or routine == "dstack":
            if len(input_size) == 2 and input_size == (1, dim):
                a.append(np.random.randint(low=0, high=100, size=(dim,)))
        run_test(tuple(a), routine, input_size)
    return


if __name__ == "__main__":
    test(10)
