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
from legate.core import LEGATE_MAX_DIM

DIM_CASES = [5, 40]


def _print_result(test_result, print_msg, err_arr):
    assert test_result, (
        f"Failed, {print_msg}\n"
        f"Attr, {err_arr[0]}\n"
        f"numpy result: {err_arr[1]}\n"
        f"cunumeric_result: {err_arr[2]}\n"
        f"cunumeric and numpy shows"
        f" different result\n"
    )
    print(f"Passed, {print_msg}")


def _broadcast_attrs(sizes):
    arr_np = list(np.arange(np.prod(size)).reshape(size) for size in sizes)
    arr_num = list(num.arange(np.prod(size)).reshape(size) for size in sizes)
    b = np.broadcast(*arr_np)
    c = num.broadcast(*arr_num)

    attrs = ["index", "nd", "ndim", "numiter", "shape", "size"]
    is_equal = True
    err_arr = None
    # test attributes
    for attr in attrs:
        if getattr(b, attr) != getattr(c, attr):
            is_equal = False
            err_arr = [attr, getattr(b, attr), getattr(c, attr)]
            break

    _print_result(is_equal, f"np.broadcast({sizes})", err_arr)


def _broadcast_value(sizes):
    arr_np = list(np.arange(np.prod(size)).reshape(size) for size in sizes)
    arr_num = list(num.arange(np.prod(size)).reshape(size) for size in sizes)
    b = np.broadcast(*arr_np)
    c = num.broadcast(*arr_num)

    is_equal = True
    err_arr = None

    # test elements in broadcasted array
    for each in zip(b, c):
        if each[0] != each[1]:
            is_equal = False
            err_arr = [("iters", b.index), each[0], each[1]]
            break
    # test reset method
    b.reset()
    c.reset()
    if b.index != c.index:
        is_equal = False
        err_arr = [("reset", b.index), each[0], each[1]]

    _print_result(is_equal, f"np.broadcast({sizes})", err_arr)


def _broadcast_view(sizes):
    arr_num = list(num.arange(np.prod(size)).reshape(size) for size in sizes)
    c = num.broadcast(*arr_num)

    is_equal = True
    err_arr = None

    # test whether the broadcast provide views of the original array
    for i in range(len(arr_num)):
        arr_num[i][(0,) * arr_num[i].ndim] = 1
        if c.iters[i][0] != arr_num[i][(0,) * arr_num[i].ndim]:
            is_equal = False
            err_arr = [
                ("view", i),
                c.iters[i][0],
                arr_num[i][(0,) * arr_num[i].ndim],
            ]

    _print_result(is_equal, f"np.broadcast({sizes})", err_arr)


def _broadcast_to_manipulation(arr, args):
    b = np.broadcast_to(*args).swapaxes(0, 1)
    c = num.broadcast_to(*args).swapaxes(0, 1)
    is_equal = True
    err_arr = [None, b, c]

    if not np.array_equal(b, c):
        is_equal = False

    _print_result(is_equal, f"np.broadcast_to({args}).swapaxes(0,1)", err_arr)


def _check(*args, params: list, routine: str):
    b = getattr(np, routine)(*args)
    c = getattr(num, routine)(*args)
    is_equal = True
    err_arr = None
    if isinstance(b, list):
        for each in zip(b, c):
            # Try to modify multiple elements in each broadcasted array
            if not np.array_equal(each[0], each[1]):
                is_equal = False
                err_arr = [("iters", b.index), each[0], each[1]]
                break

    else:
        for each in zip(b, c):
            if isinstance(each[0], np.ndarray) and not np.array_equal(
                each[0], each[1]
            ):
                is_equal = False
            elif isinstance(each[0], tuple) and each[0] != each[1]:
                is_equal = False
            if not is_equal:
                err_arr = [("value", None), each[0], each[1]]
                break

    _print_result(is_equal, f"np.{routine}({params})", err_arr)


def gen_shapes(dim):
    base = (dim,)
    result = [base]
    for i in range(1, LEGATE_MAX_DIM):
        base = base + (1,) if i % 2 == 0 else base + (dim,)
        result.append(base)
    return result


SHAPE_LISTS = {dim: gen_shapes(dim) for dim in DIM_CASES}


# test to run broadcast  w/ different size of arryas
@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_attrs(dim):
    _broadcast_attrs(SHAPE_LISTS[dim])


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_value(dim):
    _broadcast_value(SHAPE_LISTS[dim])


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_view(dim):
    _broadcast_view(SHAPE_LISTS[dim])


def test_broadcast_shapes():
    dim = DIM_CASES[0]
    shape_list = SHAPE_LISTS[dim]
    _check(*shape_list, params=shape_list, routine="broadcast_shapes")


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_to(dim):
    shape = SHAPE_LISTS[dim][-1]
    arr = np.arange(np.prod((dim,))).reshape((dim,))
    _check(arr, shape, params=(arr.shape, shape), routine="broadcast_to")


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_arrays(dim):
    shapes = SHAPE_LISTS[dim]
    arrays = list(np.arange(np.prod(shape)).reshape(shape) for shape in shapes)
    _check(*arrays, params=shapes, routine="broadcast_arrays")


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_to_mainpulation(dim):
    shape = SHAPE_LISTS[dim][-1]
    arr = np.arange(np.prod((dim,))).reshape((dim,))
    _broadcast_to_manipulation(arr, (arr.shape, shape))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
