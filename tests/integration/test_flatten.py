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
from cunumeric.array import convert_to_cunumeric_ndarray


def run_test(np_arr, num_arr):
    # We don't support 'K' yet, which will be supported later
    test_orders = ["C", "F", "A"]
    for order in test_orders:
        b = np_arr.flatten(order)
        c = num_arr.flatten(order)
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
        print_msg = f"np & cunumeric.ndarray({np_arr.shape}).flatten({order}))"
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


DIM = 10

SIZES = [
    (0,),
    (0, 10),
    (1,),
    (1, 1),
    (1, 1, 1),
    (1, DIM),
    (1, DIM, 1),
    (DIM, DIM),
    (DIM, DIM, DIM),
]


# test ndarray.flatten w/ 1D, 2D and 3D arrays
@pytest.mark.parametrize("size", SIZES, ids=str)
def test_basic(size):
    a = np.random.randint(low=0, high=100, size=size)
    b = convert_to_cunumeric_ndarray(a)
    run_test(a, b)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
