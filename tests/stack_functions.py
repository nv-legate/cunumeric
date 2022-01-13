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

import numpy as np

import cunumeric as num


def run_test(arr, test_routine):
    for routine in test_routine:
        print_msg = f"np.{routine}((array({arr[0].shape})*3)"
        b = getattr(np, routine)(arr)
        c = getattr(num, routine)(arr)
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

        assert is_equal, (
            f"Failed, {print_msg}\n"
            f"numpy result: {err_arr[0]}, {b.shape}"
            f"cunumeric_result: {err_arr[1]}, {c.shape}"
            f"cunumeric and numpy shows"
            f"different result\n"
            f"array({arr}),"
            f"routine: {routine}"
        )
        print(f"Passed, {print_msg}, np: {b.shape}, cunumeric: {c.shape}")


def test(min_size, max_size):
    if min_size >= max_size:
        max_size = min_size + 1
    print("test np.*stack routines")
    test_routine = ["vstack", "hstack", "dstack"]
    # test np.*stack w/ 1D, 2D and 3D arrays
    # test 1D arrays
    for i in range(min_size, max_size):
        a = tuple([np.random.randn(i) for num_arr in range(3)])
        run_test(a, test_routine)

        # test 2D arrays
        for j in range(min_size, max_size):
            b = tuple([np.random.randn(i, j) for num_arr in range(3)])
            run_test(b, test_routine)

            # test 3D arrays
            for k in range(min_size, max_size):
                c = tuple([np.random.randn(i, j, k) for num_arr in range(3)])
                run_test(c, test_routine)
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
        default=10,
        dest="min_size",
        help="minimum number of elements in each dimension",
    )

    args = parser.parse_args()
    test(args.min_size, args.max_size)
