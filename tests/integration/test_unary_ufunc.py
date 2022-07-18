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
import pytest
from utils.comparisons import allclose

import cunumeric as num

np.random.seed(12345)


def check_result(op, in_np, out_np, out_num):
    result = allclose(out_np, out_num) and out_np.dtype == out_num.dtype
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
    return result


def check_op(op, in_np, out_dtype="d"):
    op_np = getattr(np, op)
    op_num = getattr(num, op)

    assert op_np.nout == 1

    in_num = tuple(num.array(arr) for arr in in_np)

    out_np = op_np(*in_np)
    out_num = op_num(*in_num)

    assert check_result(op, in_np, out_np, out_num)

    out_np = np.empty(out_np.shape, dtype=out_dtype)
    out_num = num.empty(out_num.shape, dtype=out_dtype)

    op_np(*in_np, out=out_np)
    op_num(*in_num, out=out_num)

    assert check_result(op, in_np, out_np, out_num)

    out_np = np.empty(out_np.shape, dtype=out_dtype)
    out_num = num.empty(out_num.shape, dtype=out_dtype)

    op_np(*in_np, out_np)
    op_num(*in_num, out_num)

    assert check_result(op, in_np, out_np, out_num)

    # Ask cuNumeric to produce outputs to NumPy ndarrays
    out_num = np.ones(out_np.shape, dtype=out_dtype)
    op_num(*in_num, out_num)

    assert check_result(op, in_np, out_np, out_num)


def check_ops(ops, in_np, out_dtype="d"):
    for op in ops:
        check_op(op, in_np, out_dtype)


# TODO: right now we will simply check if the operations work
# for some boring inputs. For some of these, we will want to
# test corner cases in the future.


# Math operations
math_ops = (
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
)


@pytest.mark.parametrize("op", math_ops)
def test_math_ops(op):
    check_op(op, (np.random.randn(4, 5),))
    check_op(op, (np.random.randn(4, 5).astype("e"),))
    check_op(op, (np.random.randn(4, 5).astype("f"),))
    check_op(op, (np.random.randn(4, 5).astype("b"),))
    check_op(op, (np.random.randn(4, 5).astype("B"),))
    check_op(op, (np.random.randint(1, 10, size=(4, 5)),))
    check_op(op, (np.random.randn(1)[0],))


log_ops = (
    "log",
    "log10",
    "log1p",
    "log2",
)


@pytest.mark.parametrize("op", log_ops)
def test_power_ops(op):
    check_op(op, (np.random.randn(4, 5) + 3,))
    check_op(op, (np.random.randn(4, 5).astype("e") + 3,))
    check_op(op, (np.random.randn(4, 5).astype("f") + 3,))
    check_op(op, (np.random.randn(4, 5).astype("F") + 3,), out_dtype="D")
    check_op(op, (np.random.randint(3, 10, size=(4, 5)),))
    check_op(op, (np.random.randn(1)[0] + 3,))


even_root_ops = ("sqrt",)


@pytest.mark.parametrize("op", even_root_ops)
def test_even_root_ops(op):
    check_op(op, (np.random.randn(4, 5) + 3,))
    check_op(op, (np.random.randn(4, 5).astype("e") + 3,))
    check_op(op, (np.random.randn(4, 5).astype("f") + 3,))
    check_op(op, (np.random.randn(4, 5).astype("F") + 3,), out_dtype="D")
    check_op(op, (np.random.randint(3, 10, size=(4, 5)),))
    check_op(op, (np.random.randn(1)[0] + 3,))


odd_root_ops = ("cbrt",)


@pytest.mark.parametrize("op", odd_root_ops)
def test_odd_root_ops(op):
    check_op(op, (np.random.randn(4, 5),))
    check_op(op, (np.random.randn(4, 5).astype("e"),))
    check_op(op, (np.random.randn(4, 5).astype("f"),))
    check_op(op, (np.random.randint(0, 10, size=(4, 5)),))
    check_op(op, (np.random.randn(1)[0] + 3,))


trig_ops = (
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
)


@pytest.mark.parametrize("op", trig_ops)
def test_trig_ops(op):
    check_op(op, (np.random.uniform(low=-1, high=1, size=(4, 5)),))
    check_op(op, (np.random.uniform(low=-1, high=1, size=(4, 5)).astype("e"),))
    check_op(op, (np.array(np.random.uniform(low=-1, high=1)),))


arc_hyp_trig_ops = (
    "arccosh",
    "arcsinh",
)


@pytest.mark.parametrize("op", arc_hyp_trig_ops)
def test_arc_hyp_trig_ops(op):
    check_op(op, (np.random.uniform(low=1, high=5, size=(4, 5)),))
    check_op(op, (np.random.uniform(low=1, high=5, size=(4, 5)).astype("e"),))
    check_op(op, (np.array(np.random.uniform(low=1, high=5)),))


bit_ops = ("invert",)


@pytest.mark.parametrize("op", bit_ops)
def test_bit_ops(op):
    check_op(op, (np.random.randint(0, 2, size=(4, 5)),))
    check_op(op, (np.random.randint(0, 1, size=(4, 5), dtype="?"),))


comparison_ops = ("logical_not",)


@pytest.mark.parametrize("op", comparison_ops)
def test_comparison_ops(op):
    check_op(op, (np.random.randint(0, 2, size=(4, 5)),))


floating_ops = (
    "ceil",
    "floor",
    "signbit",
    # "spacing",
    "trunc",
)


@pytest.mark.parametrize("op", floating_ops)
def test_floating_ops(op):
    check_op(op, (np.random.randn(4, 5),))
    check_op(op, (np.random.randn(4, 5).astype("f"),))
    check_op(op, (np.random.randn(4, 5).astype("e"),))
    check_op(op, (np.random.randint(0, 10, size=(4, 5)),))
    check_op(op, (np.random.randint(0, 10, size=(4, 5), dtype="I"),))
    check_op(op, (np.random.randn(1)[0] + 3,))


nan_ops = (
    "isfinite",
    "isinf",
    "isnan",
    # "isnat",
)


@pytest.mark.parametrize("op", nan_ops)
def test_nan_ops(op):
    check_op(op, (np.array([-np.inf, 0.0, 1.0, np.inf, np.nan]),))
    check_op(op, (np.array(np.inf),))


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
        default="1",
        help="input data",
    )
    parser.add_argument(
        "--dtypes",
        dest="dtypes",
        default="l",
        help="input data",
    )
    args, extra = parser.parse_known_args()

    sys.argv = sys.argv[:1] + extra

    if args.op is not None:
        in_np = parse_inputs(args.inputs, args.dtypes)
        check_ops([args.op], in_np)
    else:
        sys.exit(pytest.main(sys.argv))
