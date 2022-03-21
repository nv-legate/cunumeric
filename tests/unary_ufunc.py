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


def test(ops, in_np, out_dtype="d"):
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


def test_all_unary_ops():
    # TODO: right now we will simply check if the operations work
    # for some boring inputs. For some of these, we will want to
    # test corner cases in the future.

    np.random.seed(12345)

    # Math operations
    ops = [
        "absolute",
        "conjugate",
        "exp",
        "exp2",
        "expm1",
        "fabs",
        "logical_not",
        "negative",
        "positive",
        "reciprocal",
        "rint",
        "sign",
        "square",
    ]
    test(ops, (np.random.randn(4, 5),))
    test(ops, (np.random.randn(4, 5).astype("e"),))
    test(ops, (np.random.randn(4, 5).astype("f"),))
    test(ops, (np.random.randint(1, 10, size=(4, 5)),))
    test(ops, (np.random.randn(1)[0],))

    ops = [
        "log",
        "log10",
        "log1p",
        "log2",
        "sqrt",
    ]
    test(ops, (np.random.randn(4, 5) + 3,))
    test(ops, (np.random.randn(4, 5).astype("e") + 3,))
    test(ops, (np.random.randn(4, 5).astype("f") + 3,))
    test(ops, (np.random.randn(4, 5).astype("F") + 3,), out_dtype="D")
    test(ops, (np.random.randint(3, 10, size=(4, 5)),))
    test(ops, (np.random.randn(1)[0] + 3,))

    ops = [
        "cbrt",
    ]
    test(ops, (np.random.randn(4, 5),))
    test(ops, (np.random.randn(4, 5).astype("e"),))
    test(ops, (np.random.randn(4, 5).astype("f"),))
    test(ops, (np.random.randint(0, 10, size=(4, 5)),))
    test(ops, (np.random.randn(1)[0] + 3,))

    # Trigonometric functions
    ops = [
        "arccos",
        "arcsin",
        "arctan",
        "arctanh",
        "cos",
        "cosh",
        "deg2rad",
        "rad2deg",
        "sin",
        "sinh",
        "tan",
        "tanh",
    ]
    test(ops, (np.random.uniform(low=-1, high=1, size=(4, 5)),))
    test(ops, (np.random.uniform(low=-1, high=1, size=(4, 5)).astype("e"),))
    test(ops, (np.array(np.random.uniform(low=-1, high=1)),))

    ops = [
        "arccosh",
        "arcsinh",
    ]
    test(ops, (np.random.uniform(low=1, high=5, size=(4, 5)),))
    test(ops, (np.random.uniform(low=1, high=5, size=(4, 5)).astype("e"),))
    test(ops, (np.array(np.random.uniform(low=1, high=5)),))

    # Bit-twiddling functions
    ops = [
        "invert",
    ]
    test(ops, (np.random.randint(0, 2, size=(4, 5)),))
    test(ops, (np.random.randint(0, 1, size=(4, 5), dtype="?"),))

    # Comparison functions
    ops = [
        "logical_not",
    ]
    test(ops, (np.random.randint(0, 2, size=(4, 5)),))

    # Floating functions

    ops = [
        "ceil",
        "floor",
        # "fmod",
        # "frexp",
        # "modf",
        "signbit",
        # "spacing",
        "trunc",
    ]
    test(ops, (np.random.randn(4, 5),))
    test(ops, (np.random.randn(4, 5).astype("f"),))
    test(ops, (np.random.randn(4, 5).astype("e"),))
    test(ops, (np.random.randint(0, 10, size=(4, 5)),))
    test(ops, (np.random.randint(0, 10, size=(4, 5), dtype="I"),))
    test(ops, (np.random.randn(1)[0] + 3,))

    ops = [
        "isfinite",
        "isinf",
        "isnan",
        # "isnat",
    ]
    test(ops, (np.array([-np.inf, 0.0, 1.0, np.inf, np.nan]),))
    test(ops, (np.array(np.inf),))


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
        default="1",
        help="input data",
    )
    parser.add_argument(
        "--dtypes",
        dest="dtypes",
        default="l",
        help="input data",
    )
    args, unknown = parser.parse_known_args()

    if args.op is not None:
        in_np = parse_inputs(args.inputs, args.dtypes)
        test([args.op], in_np)
    else:
        test_all_unary_ops()
