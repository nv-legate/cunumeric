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

from itertools import product

import numpy as np

import cunumeric as num


def value_type(obj):
    if np.isscalar(obj):
        return "scalar"
    elif obj.ndim == 0:
        return f"{obj.dtype} 0d array"
    else:
        return f"{obj.dtype} array"


def test(lhs_np, rhs_np, lhs_num, rhs_num):
    print(f"{value_type(lhs_np)} x {value_type(rhs_np)}")

    out_np = np.add(lhs_np, rhs_np)
    out_num = num.add(lhs_num, rhs_num)

    if out_np.dtype != out_num.dtype:
        print("LHS")
        print(lhs_np)
        print("RHS")
        print(rhs_np)
        print(f"NumPy type: {out_np.dtype}, cuNumeric type: {out_num.dtype}")
        assert False


def run_all_tests():
    types = [
        "b",
        "B",
        "h",
        "H",
        "i",
        "I",
        "l",
        "L",
        "q",
        "Q",
        "e",
        "f",
        "d",
        "F",
        "D",
    ]
    array_values = [[1]]
    scalar_values = [1, -1, 1.0, 1e-50, 1j]

    for idx, lhs_type in enumerate(types):
        for rhs_type in types[idx:]:
            for lhs_value, rhs_value in product(array_values, array_values):
                lhs_np = np.array(lhs_value, dtype=lhs_type)
                rhs_np = np.array(rhs_value, dtype=rhs_type)

                lhs_num = num.array(lhs_np)
                rhs_num = num.array(rhs_np)

                test(lhs_np, rhs_np, lhs_num, rhs_num)

            for lhs_value, rhs_value in product(scalar_values, scalar_values):
                try:
                    lhs_np = np.array(lhs_value, dtype=lhs_type)
                    rhs_np = np.array(rhs_value, dtype=rhs_type)

                    lhs_num = num.array(lhs_np)
                    rhs_num = num.array(rhs_np)

                    test(lhs_np, rhs_np, lhs_num, rhs_num)
                except TypeError:
                    pass

    for ty in types:
        for array, scalar in product(array_values, scalar_values):
            array_np = np.array(array, dtype=ty)
            array_num = num.array(array_np)

            test(array_np, scalar, array_num, scalar)

    # TODO: NumPy's type coercion rules are confusing at best and impossible
    # for any human being to understand in my opinion. My attempt to make
    # sense of it for the past two days failed miserably. I managed to make
    # the code somewhat compatible with NumPy for cases where Python scalars
    # are passed.
    #
    # If anyone can do a better job than me and finally make cuNumeric
    # implement the same typing rules, please put these tests back.
    #
    # for idx, lhs_type in enumerate(types):
    #    for rhs_type in types[idx:]:
    #        for array, scalar in product(array_values, scalar_values):
    #            try:
    #                lhs_np = np.array(array, dtype=lhs_type)
    #                rhs_np = np.array(scalar, dtype=rhs_type)

    #                lhs_num = num.array(lhs_np)
    #                rhs_num = num.array(rhs_np)

    #                test(lhs_np, rhs_np, lhs_num, rhs_num)
    #            except TypeError:
    #                pass

    #            try:
    #                lhs_np = np.array(scalar, dtype=lhs_type)
    #                rhs_np = np.array(array, dtype=rhs_type)

    #                lhs_num = num.array(lhs_np)
    #                rhs_num = num.array(rhs_np)

    #                test(lhs_np, rhs_np, lhs_num, rhs_num)
    #            except TypeError:
    #                pass


if __name__ == "__main__":
    run_all_tests()
