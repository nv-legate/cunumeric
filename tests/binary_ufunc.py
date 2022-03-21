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

import argparse
from itertools import product

import numpy as np

import cunumeric as num


def check_result(op, in_np, out_np, out_num):
    result = np.allclose(out_np, out_num) and out_np.dtype == out_num.dtype
    if not result:
        print(f"cunumeric.{op} failed the test")
        print("Inputs:")
        for arr in in_np:
            print(arr)
            print(f"dtype: {arr.dtype}")
        print("NumPy output:")
        print(out_np)
        print(f"dtype: {out_np.dtype}")
        print("cuNumeric output:")
        print(out_num)
        print(f"dtype: {out_num.dtype}")
        assert False


def test(ops, in_np, out_dtype="D"):
    for op in ops:
        op_np = getattr(np, op)
        op_num = getattr(num, op)

        assert op_np.nout == 1

        in_num = tuple(num.array(arr) for arr in in_np)

        out_np = op_np(*in_np)
        out_num = op_num(*in_num)

        check_result(op, in_np, out_np, out_num)

        out_np = np.empty(out_np.shape, dtype=out_dtype)
        out_num = num.empty(out_num.shape, dtype=out_dtype)

        op_np(*in_np, out=out_np)
        op_num(*in_num, out=out_num)

        check_result(op, in_np, out_np, out_num)


def test_all_binary_ops():
    # TODO: right now we will simply check if the operations work
    # for some boring inputs. For some of these, we will want to
    # test corner cases in the future.

    np.random.seed(12345)

    # Math operations
    ops = [
        "add",
        # "arctan2",
        # "bitwise_and",
        # "bitwise_or",
        # "bitwise_xor",
        # "copysign",
        # "divmod",
        "equal",
        # "float_power",
        # "fmax",
        # "fmin",
        # "fmod",
        # "gcd",
        "greater",
        "greater_equal",
        # "heaviside",
        # "hypot",
        # "lcm",
        # "ldexp",
        # "left_shift",
        "less",
        "less_equal",
        # "logaddexp",
        # "logaddexp2",
        "logical_and",
        "logical_or",
        "logical_xor",
        "maximum",
        "minimum",
        "multiply",
        # "nextafter",
        "not_equal",
        # "right_shift",
        "subtract",
        "true_divide",
    ]

    # We want to test array-array, array-scalar, and scalar-array cases
    arrs = (
        np.random.randint(3, 10, size=(4, 5)).astype("I"),
        np.random.uniform(size=(4, 5)).astype("F"),
    )
    scalars = (
        np.uint64(2),
        np.int64(-3),
        np.random.randn(1)[0],
        np.complex64(1 + 1j),
    )

    for arr1, arr2 in product(arrs, arrs):
        test(ops, (arr1, arr2))

    for arr, scalar in product(arrs, scalars):
        test(ops, (arr, arr))
        test(ops, (scalar, arr))

    for scalar1, scalar2 in product(scalars, scalars):
        test(ops, (scalar1, scalar2))

    ops = [
        "floor_divide",
    ]

    for arr1, arr2 in product(arrs[:-1], arrs[:-1]):
        test(ops, (arr1, arr2))

    for arr, scalar in product(arrs[:-1], scalars[:-1]):
        test(ops, (arr, arr))
        test(ops, (scalar, arr))

    for scalar1, scalar2 in product(scalars[:-1], scalars[:-1]):
        test(ops, (scalar1, scalar2))

    ops = [
        "power",
    ]

    for arr1, arr2 in product(arrs, arrs):
        test(ops, (arr1, arr2))

    for arr, scalar in product(arrs, scalars[:1]):
        test(ops, (arr, arr))
        test(ops, (scalar, arr))

    for scalar1, scalar2 in product(scalars[:1], scalars[:1]):
        test(ops, (scalar1, scalar2))

    ops = [
        "remainder",
    ]

    for arr1, arr2 in product(arrs[:-2], arrs[:-2]):
        test(ops, (arr1, arr2))

    for arr, scalar in product(arrs[:-2], scalars[:-2]):
        test(ops, (arr, arr))
        test(ops, (scalar, arr))

    for scalar1, scalar2 in product(scalars[:-2], scalars[:-2]):
        test(ops, (scalar1, scalar2))


def parse_inputs(in_str, dtype_str):
    dtypes = tuple(np.dtype(dtype) for dtype in dtype_str.split(":"))
    tokens = in_str.split(":")
    inputs = []
    for token, dtype in zip(tokens, dtypes):
        split = token.split(",")
        if len(split) == 1:
            inputs.append(dtype.type(split[0]))
        else:
            inputs.append(np.array(split, dtype=dtype))
    return inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opname",
        default=None,
        dest="op",
        help="the name of operation to test",
    )
    parser.add_argument(
        "--inputs",
        dest="inputs",
        default="1:1",
        help="input data",
    )
    parser.add_argument(
        "--dtypes",
        dest="dtypes",
        default="l:l",
        help="input data",
    )
    args, unknown = parser.parse_known_args()

    if args.op is not None:
        in_np = parse_inputs(args.inputs, args.dtypes)
        test([args.op], in_np)
    else:
        test_all_binary_ops()
