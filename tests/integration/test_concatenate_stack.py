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
import pytest

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
                if not np.array_equal(*each):
                    err_arr = each
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


DIM = 10

SIZES = [
    (0,),
    (0, 10),
    (1,),
    (1, 1),
    (1, 1, 1),
    (1, DIM),
    (DIM, DIM),
    (DIM, DIM, DIM),
]


@pytest.fixture(autouse=True)
def a(size):
    return [np.random.randint(low=0, high=100, size=size) for _ in range(3)]


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_concatenate(size, a):
    run_test(tuple(a), "concatenate", size)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_stack(size, a):
    run_test(tuple(a), "stack", size)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_hstack(size, a):
    run_test(tuple(a), "hstack", size)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_column_stack(size, a):
    run_test(tuple(a), "column_stack", size)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_column_vstack(size, a):
    # exception for 1d array on vstack
    if len(size) == 2 and size == (1, DIM):
        a.append(np.random.randint(low=0, high=100, size=(DIM,)))
    run_test(tuple(a), "vstack", size)


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_column_dstack(size, a):
    # exception for 1d array on dstack
    if len(size) == 2 and size == (1, DIM):
        a.append(np.random.randint(low=0, high=100, size=(DIM,)))
    run_test(tuple(a), "dstack", size)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
