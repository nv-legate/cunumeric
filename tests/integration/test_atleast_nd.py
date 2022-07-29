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
from legate.core import LEGATE_MAX_DIM

import cunumeric as num


def _check(a, routine, sizes):
    b = getattr(np, routine)(*a)
    c = getattr(num, routine)(*a)
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
    print_msg = f"np.{routine}({sizes})"
    assert is_equal, (
        f"Failed, {print_msg}\n"
        f"numpy result: {err_arr[0]}\n"
        f"cunumeric_result: {err_arr[1]}\n"
        f"cunumeric and numpy shows different result\n"
    )
    print(f"Passed, {print_msg}, np: {b}, cunumeric: {c}")


DIM = 10

SIZE_CASES = list((DIM,) * ndim for ndim in range(LEGATE_MAX_DIM + 1))

SIZE_CASES += [
    (0,),  # empty array
    (1,),  # singlton array
]


# test to run atleast_nd w/ a single array
@pytest.mark.parametrize("size", SIZE_CASES, ids=str)
def test_atleast_1d(size):
    a = [np.arange(np.prod(size)).reshape(size)]
    _check(a, "atleast_1d", size)


@pytest.mark.parametrize("size", SIZE_CASES, ids=str)
def test_atleast_2d(size):
    a = [np.arange(np.prod(size)).reshape(size)]
    _check(a, "atleast_2d", size)


@pytest.mark.parametrize("size", SIZE_CASES, ids=str)
def test_atleast_3d(size):
    a = [np.arange(np.prod(size)).reshape(size)]
    _check(a, "atleast_3d", size)


# test to run atleast_nd w/ list of arrays
@pytest.mark.parametrize("dim", range(1, 4))
def test_atleast_nd(dim):
    a = list(np.arange(np.prod(size)).reshape(size) for size in SIZE_CASES)
    _check(a, f"atleast_{dim}d", SIZE_CASES)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
