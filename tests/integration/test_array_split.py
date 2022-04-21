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

import math

import numpy as np
import pytest

import cunumeric as num

# Seed the random generator with a random number
np.random.seed(416)

DIM = 20

# test the array_split routines on empty, singleton, 2D and 3D arrays
# w/ integers, list of indicies. vsplit, hsplit, dsplit are included
# in the following loops(axis = 0: vsplit, 1: hsplit, 2: dsplit)
SIZES = [
    (0,),
    (0, 10),
    (1),
    (1, 1),
    (1, 1, 1),
    (DIM, DIM),
    (DIM, DIM, DIM),
]


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_array_split(size):

    a = np.random.randint(low=0, high=100, size=size)
    for axis in range(a.ndim):
        input_arr = []
        even_div = None
        uneven_div = None

        if a.shape[axis] > 1:
            for div in range(1, (int)(math.sqrt(a.shape[axis]) + 1)):
                if a.shape[axis] % div == 0:
                    even_div = div
                else:
                    uneven_div = div
                if even_div is not None and uneven_div is not None:
                    break
        else:
            even_div = 2
            uneven_div = 1

        # divisible integer
        input_arr.append(even_div)
        # indivisble integer
        input_arr.append(uneven_div)
        # integer larger than shape[axis]
        input_arr.append(a.shape[axis] + np.random.randint(1, 10))
        # indices array which has points
        # within the target dimension of the src array
        if a.shape[axis] > 1:
            input_arr.append(list(range(1, a.shape[axis], even_div)))
        # indices array which has points
        # out of the target dimension of the src array
        input_arr.append(
            list(
                range(
                    0,
                    a.shape[axis] + even_div * np.random.randint(1, 10),
                    even_div,
                )
            )
        )
        input_arr.append(
            list(
                range(
                    a.shape[axis] + even_div * np.random.randint(1, 10),
                    0,
                    -even_div,
                )
            )
        )

        for input_opt in input_arr:
            # test divisible integer or indices
            print_msg = f"np.array_split({a.shape}, {input_opt}" f", {axis})"
            # Check if both impls produce the error
            # for non-viable options
            b = np.array_split(a, input_opt, axis)
            c = num.array_split(a, input_opt, axis)
            is_equal = True
            err_arr = [b, c]

            if len(b) != len(c):
                is_equal = False
                err_arr = [b, c]
            else:
                for each in zip(b, c):
                    if not np.array_equal(each[0], each[1]):
                        err_arr = each
                        is_equal = False
                        break

            assert is_equal, (
                f"Failed, {print_msg}"
                f"numpy result: {err_arr[0]}"
                f"cunumeric_result: {err_arr[1]}"
                f"cunumeric and numpy shows"
                f"different result\n"
                f"array({size}),"
                f"routine: array_split,"
                f"indices: {input_opt}, axis: {axis}"
            )

            print(f"Passed, {print_msg}")


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
