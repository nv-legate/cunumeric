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
import pytest
from utils.comparisons import allclose

import cunumeric as num


def check_result(op, in_np, out_np, out_num):
    rtol = 1e-02 if any(x.dtype == np.float16 for x in in_np) else 1e-05
    result = (
        allclose(out_np, out_num, rtol=rtol) and out_np.dtype == out_num.dtype
    )
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


def check_ops(ops, in_np, out_dtype="D"):
    in_num = tuple(num.array(arr) for arr in in_np)

    for op in ops:
        if op.isidentifier():
            op_np = getattr(np, op)
            op_num = getattr(num, op)
            assert op_np.nout == 1

            out_np = op_np(*in_np)
            out_num = op_num(*in_num)

            check_result(op, in_np, out_np, out_num)

            out_np = np.empty(out_np.shape, dtype=out_dtype)
            out_num = num.empty(out_num.shape, dtype=out_dtype)
            op_np(*in_np, out=out_np)
            op_num(*in_num, out=out_num)

            check_result(op, in_np, out_np, out_num)

            # Ask cuNumeric to produce outputs to NumPy ndarrays
            out_num = np.empty(out_np.shape, dtype=out_dtype)
            op_num(*in_num, out=out_num)

            check_result(op, in_np, out_np, out_num)

        else:
            # Doing it this way instead of invoking the dunders directly, to
            # avoid having to select the right version, __add__ vs __radd__,
            # when one isn't supported, e.g. for scalar.__add__(array)

            out_np = eval(f"in_np[0] {op} in_np[1]")
            out_num = eval(f"in_num[0] {op} in_num[1]")

            check_result(op, in_np, out_np, out_num)

            out_np = np.ones_like(out_np)
            out_num = num.ones_like(out_num)
            exec(f"out_np {op}= in_np[0]")
            exec(f"out_num {op}= in_num[0]")

            check_result(op, in_np, out_np, out_num)

            out_num = np.ones_like(out_np)
            exec(f"out_num {op}= in_num[0]")

            check_result(op, in_np, out_np, out_num)


def test_all():
    # TODO: right now we will simply check if the operations work
    # for some boring inputs. For some of these, we will want to
    # test corner cases in the future.

    # TODO: matmul, @

    # Math operations
    ops = [
        "*",
        "+",
        "-",
        "/",
        "add",
        # "divmod",
        "equal",
        "fmax",
        "fmin",
        "greater",
        "greater_equal",
        # "heaviside",
        # "ldexp",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "maximum",
        "minimum",
        "multiply",
        "not_equal",
        "subtract",
        "true_divide",
    ]

    # We want to test array-array, array-scalar, and scalar-array cases
    arrs = (
        np.random.randint(3, 10, size=(4, 5)).astype("I"),
        np.random.uniform(size=(4, 5)).astype("e"),
        np.random.uniform(size=(4, 5)).astype("f"),
        np.random.uniform(size=(4, 5)).astype("d"),
        np.random.uniform(size=(4, 5)).astype("F"),
    )

    scalars = (
        np.uint64(2),
        np.int64(-3),
        np.random.randn(1)[0],
        np.complex64(1 + 1j),
    )

    for arr1, arr2 in product(arrs, arrs):
        check_ops(ops, (arr1, arr2))

    for arr, scalar in product(arrs, scalars):
        check_ops(ops, (arr, scalar))
        check_ops(ops, (scalar, arr))

    for scalar1, scalar2 in product(scalars, scalars):
        check_ops(ops, (scalar1, scalar2))

    ops = [
        "//",
        "arctan2",
        "copysign",
        "floor_divide",
        "mod",
        "fmod",
        "hypot",
        "logaddexp",
        "logaddexp2",
        "nextafter",
    ]

    for arr1, arr2 in product(arrs[:-1], arrs[:-1]):
        check_ops(ops, (arr1, arr2))

    for arr, scalar in product(arrs[:-1], scalars[:-1]):
        check_ops(ops, (arr, scalar))
        check_ops(ops, (scalar, arr))

    for scalar1, scalar2 in product(scalars[:-1], scalars[:-1]):
        check_ops(ops, (scalar1, scalar2))

    ops = [
        "**",
        "power",
        "float_power",
    ]

    for arr1, arr2 in product(arrs, arrs):
        check_ops(ops, (arr1, arr2))

    for arr in arrs:
        check_ops(ops, (arr, scalars[0]))
        check_ops(ops, (scalars[0], arr))
        check_ops(ops, (arr, scalars[3]))
        check_ops(ops, (scalars[3], scalars[3]))

    check_ops(ops, (scalars[0], scalars[3]))
    check_ops(ops, (scalars[3], scalars[0]))

    ops = [
        "%",
        "remainder",
    ]

    for arr1, arr2 in product(arrs[:1], arrs[:1]):
        check_ops(ops, (arr1, arr2))

    for arr, scalar in product(arrs[:1], scalars[:-2]):
        check_ops(ops, (arr, scalar))
        check_ops(ops, (scalar, arr))

    for scalar1, scalar2 in product(scalars[:-2], scalars[:-2]):
        check_ops(ops, (scalar1, scalar2))

    ops = [
        "&",
        "<<",
        ">>",
        "^",
        "|",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "gcd",
        "lcm",
        "left_shift",
        "right_shift",
    ]

    check_ops(ops, (arr1[0], arr2[0]))

    check_ops(ops, (arrs[0], scalars[0]))
    check_ops(ops, (arrs[0], scalars[1]))
    check_ops(ops, (scalars[0], arrs[0]))
    check_ops(ops, (scalars[1], arrs[0]))

    check_ops(ops, (scalars[0], scalars[0]))


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
    import sys

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
    args, extra = parser.parse_known_args()

    sys.argv = sys.argv[:1] + extra

    if args.op is not None:
        in_np = parse_inputs(args.inputs, args.dtypes)
        check_ops([args.op], in_np)
    else:
        sys.exit(pytest.main(sys.argv))
