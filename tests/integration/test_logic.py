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

from itertools import combinations_with_replacement

import numpy as np
import pytest

import cunumeric as num


def test_inf():
    arr = [np.inf, -np.inf, np.nan, 0]
    assert np.array_equal(np.isposinf(arr), num.isposinf(arr))
    assert np.array_equal(np.isneginf(arr), num.isneginf(arr))


INPUTS = (
    [1.0, 2.0, 3.0],
    [1.0 + 0j, 2.0 + 0j, 3.0 + 0j],
    [1.0 + 1j, 2.0 + 0j, 3.0 + 1j],
)


@pytest.mark.parametrize("arr", INPUTS)
def test_predicates(arr):
    in_np = np.array(arr)
    in_num = num.array(arr)

    assert np.array_equal(np.isreal(in_np), num.isreal(in_num))
    assert np.array_equal(np.iscomplex(in_np), num.iscomplex(in_num))


@pytest.mark.parametrize("arr", INPUTS)
def test_array_predicates(arr):
    in_np = np.array(arr)
    in_num = num.array(arr)

    assert np.array_equal(np.isrealobj(in_np), num.isrealobj(in_num))
    assert np.array_equal(np.isrealobj(in_np), num.isrealobj(in_np))
    assert np.array_equal(np.iscomplexobj(in_np), num.iscomplexobj(in_num))
    assert np.array_equal(np.iscomplexobj(in_np), num.iscomplexobj(in_np))


def test_isscalar():
    in_np = np.array([1, 2, 3])
    in_num = num.array([1, 2, 3])

    assert num.isscalar(1.0)
    assert num.isscalar(True)
    assert not num.isscalar(in_np)
    assert not num.isscalar(in_num)

    # NumPy's scalar reduction returns a Python scalar
    assert num.isscalar(np.sum(in_np))
    # but cuNumeric's scalar reduction returns a 0-D array that behaves like
    # a deferred scalar
    assert not num.isscalar(num.sum(in_np))


def test_isclose():
    in1_np = np.random.rand(10)
    in2_np = in1_np + np.random.uniform(low=5e-09, high=2e-08, size=10)
    in1_num = num.array(in1_np)
    in2_num = num.array(in2_np)

    out_np = np.isclose(in1_np, in2_np)
    out_num = num.isclose(in1_num, in2_num)
    assert np.array_equal(out_np, out_num)

    weird_values = [np.inf, -np.inf, np.nan, 0.0, -0.0]
    weird_pairs = tuple(combinations_with_replacement(weird_values, 2))
    in1_np = np.array([x for x, _ in weird_pairs])
    in2_np = np.array([y for _, y in weird_pairs])
    in1_num = num.array(in1_np)
    in2_num = num.array(in2_np)

    out_np = np.isclose(in1_np, in2_np)
    out_num = num.isclose(in1_num, in2_num)
    assert np.array_equal(out_np, out_num)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
