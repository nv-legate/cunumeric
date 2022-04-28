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


def _deepen(depth, x):
    for _ in range(depth):
        x = [x]
    return x


def _check(a, b, depth, sizes):
    arr = [_deepen(depth, a), _deepen(depth, b)]
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
    print_msg = f"np.block([{_deepen(depth, sizes)}, {_deepen(depth, sizes)}])"
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


DIM = 10

SIZE_CASES = [
    [(0,), (0,)],  # empty arrays
    [(1,), (1,)],  # singlton arrays
    [(DIM, 1), (DIM, DIM)],  # 1D and 2D arrays
    [(DIM, 1), (DIM, 1), (DIM, DIM)],  # 3 arrays in the inner-most list
]


@pytest.mark.parametrize("sizes", SIZE_CASES, ids=str)
@pytest.mark.parametrize("depth", range(3))
def test(depth, sizes):
    a = [np.arange(np.prod(size)).reshape(size) for size in sizes]
    b = [np.arange(np.prod(size)).reshape(size) for size in sizes]
    _check(a, b, depth, sizes)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
