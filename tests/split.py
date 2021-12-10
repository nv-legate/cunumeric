# Copyright 2021 NVIDIA Corporation
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

import argparse
import itertools
import math

import numpy as np

import cunumeric as num


def test(min_size, max_size):
    if min_size >= max_size:
        max_size = min_size + 1
    # Seed the random generator with a random number
    np.random.seed(416)
    print("test split")
    test_routine = ["array_split", "split"]
    # test the split routines on 3D array [1:10, 1:10, 1:10] w/ integers,
    # list of indicies. vsplit, hsplit, dsplit are included
    # in the following loops(axis = 0: vsplit, 1: hsplit, 2: dsplit)
    for i in range(min_size, max_size):
        for j in range(min_size, max_size):
            for k in range(min_size, max_size):
                a = num.random.randn(i, j, k)
                for axis in range(a.ndim):
                    input_arr = []
                    even_div = None  # a.shape[axis]
                    uneven_div = None  # a.shape[axis]
                    for div in range(1, (int)(math.sqrt(a.shape[axis]) + 1)):
                        if a.shape[axis] % div == 0:
                            even_div = div
                        else:
                            uneven_div = div
                        if even_div is not None and uneven_div is not None:
                            break
                    if even_div is None:
                        even_div = a.shape[axis]
                    # divisible integer
                    input_arr.append(even_div)
                    # indivisble integer
                    if even_div != uneven_div and uneven_div is not None:
                        input_arr.append(uneven_div)
                    # indices array which has points
                    # within the target dimension of the src array
                    if a.shape[axis] > 1:
                        input_arr.append(
                            list(range(1, a.shape[axis], even_div))
                        )
                    # indices array which has points
                    # out of the target dimension of the src array
                    input_arr.append(
                        list(
                            range(
                                0,
                                a.shape[axis]
                                + even_div * np.random.randint(1, 10),
                                even_div,
                            )
                        )
                    )
                    input_arr.append(
                        list(
                            range(
                                a.shape[axis]
                                + even_div * np.random.randint(1, 10),
                                0,
                                -even_div,
                            )
                        )
                    )

                    for routine, input_opt in itertools.product(
                        test_routine, input_arr
                    ):
                        # test divisible integer or indices
                        print(
                            "Testing np.{}({}, {}, {}) -> ".format(
                                routine, a.shape, input_opt, axis
                            ),
                            end="",
                        )
                        b_result = True
                        c_result = True
                        # Check if both impls produce the error
                        # for non-viable options
                        try:
                            b = getattr(np, routine)(a, input_opt, axis)
                        except ValueError:
                            b_result = False
                        try:
                            c = getattr(num, routine)(a, input_opt, axis)
                        except ValueError:
                            c_result = False
                        if b_result and c_result:
                            for each in zip(b, c):
                                sub_set = list(each)
                                if not np.array_equal(sub_set[0], sub_set[1]):
                                    print(" Failed")
                                    print(a)
                                    print(
                                        "numpy result: {}".format(sub_set[0])
                                    )
                                    print(
                                        "cunumeric result: {}".format(
                                            sub_set[1]
                                        )
                                    )
                                    raise ValueError(
                                        (
                                            "cunumeric and numpy shows "
                                            "different result\n"
                                            "array({},{},{}), "
                                            "routine: {}, "
                                            "indices: {}, axis: {}"
                                        ).format(
                                            i, j, k, routine, input_opt, axis
                                        )
                                    )
                        print(" Passed")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--max",
        type=int,
        default=10,
        dest="max_size",
        help="maximum number of elements in each dimension",
    )
    parser.add_argument(
        "-b",
        "--min",
        type=int,
        default=1,
        dest="min_size",
        help="minimum number of elements in each dimension",
    )

    args = parser.parse_args()
    test(args.min_size, args.max_size)
