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


def _broadcast_check(a, sizes):
    b = np.broadcast(*a)
    c = num.broadcast(*a)

    attrs = [
        attr
        for attr in dir(b)
        if not attr.startswith("___") and attr != "iters"
    ]

    is_equal = True

    err_arr = None
    # test attributes
    for attr in attrs:
        if getattr(b, attr) != getattr(c, attr):
            is_equal = False
            err_arr = [attr, getattr(b, attr), getattr(c, attr)]

    # test elements in broadcasted array
    for each in zip(b, c):
        if each[0] != each[1]:
            is_equal = False
            err_arr = [("iters", b.index), each[0], each[1]]
            break

    print_msg = f"np.broadcast({sizes})"
    assert is_equal, (
        f"Failed, {print_msg}\n"
        f"Attr, {err_arr[0]}\n"
        f"numpy result: {err_arr[1]}\n"
        f"cunumeric_result: {err_arr[2]}\n"
        f"cunumeric and numpy shows"
        f" different result\n"
    )

    print(f"Passed, {print_msg}")


DIM = 10

SIZE_CASES = list((DIM,) * ndim for ndim in range(LEGATE_MAX_DIM + 1))


# test to run atleast_nd w/ list of arrays
def test_broadcast():
    a = list(np.arange(np.prod(size)).reshape(size) for size in SIZE_CASES)
    _broadcast_check(a, SIZE_CASES)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
