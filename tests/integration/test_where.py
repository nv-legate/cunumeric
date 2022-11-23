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

import numpy as np
import pytest
from utils.generators import mk_seq_array

import cunumeric as num

CONDITIONS = [
    [[True, False], [True, True]],
    [[True, False]],
    [True, False],
    False,
    [[0.0, 1.0], [0, -2]],
]


def test_basic():
    anp = np.array([1, 54, 4, 4, 0, 45, 5, 58, 0, 9, 0, 4, 0, 0, 0, 5, 0])
    a = num.array(anp)
    assert num.array_equal(np.where(anp), num.where(a))


@pytest.mark.parametrize("cond", CONDITIONS, ids=str)
def test_condition(cond):
    anp = np.array(cond)
    xnp = np.array([[1, 2], [3, 4]])
    ynp = np.array([[9, 8], [7, 6]])
    a = num.array(anp)
    x = num.array(xnp)
    y = num.array(ynp)
    assert np.array_equal(np.where(anp, xnp, ynp), num.where(a, x, y))


@pytest.mark.parametrize(
    "shape_a",
    ((1,), (3,), (1, 3), (3, 3), (2, 3, 3)),
    ids=lambda shape_a: f"(shape_a={shape_a})",
)
def test_broadcast(shape_a):
    a = mk_seq_array(num, shape_a)
    anp = mk_seq_array(np, shape_a)
    cond = a > 5
    cond_np = anp > 5

    shape_x = (3, 3)
    x = mk_seq_array(num, shape_x)
    xnp = mk_seq_array(np, shape_x)
    shape_y = (1, 3)
    y = mk_seq_array(num, shape_y) * 10
    ynp = mk_seq_array(np, shape_y) * 10

    assert np.array_equal(np.where(cond_np, xnp, ynp), num.where(cond, x, y))


@pytest.mark.xfail
def test_condition_none():
    # In Numpy, pass and returns [1, 2]
    # In cuNumeric, raises AttributeError:
    # 'NoneType' object has no attribute '_maybe_convert'
    x = 0
    ynp = np.array([1, 2])
    y = num.array(ynp)
    assert np.array_equal(np.where(None, x, ynp), num.where(None, x, y))


@pytest.mark.xfail
@pytest.mark.parametrize(
    "values",
    ((None, None), (None, 1), (1, None)),
    ids=lambda values: f"(values={values})",
)
def test_x_y_none(values):
    # For x=None and y=None,
    # In Numpy, pass and returns [None, None]
    # In cuNumeric, pass and returns (array([0]),)
    # For x=None and y=1
    # In Numpy, pass and returns [None, 1]
    # In cuNumeric, raises ValueError: both 'x' and 'y' parameters
    # must be specified together for where
    cond = [True, False]
    anp = np.array(cond)
    a = num.array(anp)
    x, y = values
    assert np.array_equal(np.where(anp, x, y), num.where(a, x, y))


def test_x_y_type():
    xnp = np.arange(4, dtype=np.int32)
    ynp = np.arange(4, dtype=np.float32) * 2.2
    x = num.array(xnp)
    y = num.array(ynp)

    res_np = np.where(xnp > 2.0, xnp, ynp)
    res = num.where(x > 2.0, x, y)

    assert np.array_equal(res_np, res)
    assert res_np.dtype == res.dtype


def test_condition_empty():
    cond = num.array([])
    cond_np = np.array([])
    x = 0
    y = 1
    assert np.array_equal(np.where(cond_np, x, y), num.where(cond, x, y))


class TestWhereErrors:
    @pytest.mark.parametrize(
        "shape_y",
        ((0,), (2,), (1, 2), (4, 1)),
        ids=lambda shape_y: f"(shape_y={shape_y})",
    )
    def test_x_y_bad_shape(self, shape_y):
        shape_a = (3, 3)
        a = mk_seq_array(num, shape_a)
        cond = a > 5
        x = 1
        y = mk_seq_array(num, shape_y)

        msg = "shape mismatch"
        with pytest.raises(ValueError, match=msg):
            num.where(cond, x, y)


INPUT = [
    [1, 54, 4, 4, 0, 45, 5, 58, 0, 9, 0, 4, 0, 0, 0, 5, 0, 1],
    [[1, 54, 4], [4, 0, 45], [5, 58, 0], [9, 0, 4], [0, 0, 0], [5, 0, 1]],
    [
        [[1, 54, 4], [4, 0, 45]],
        [[5, 58, 0], [9, 0, 4]],
        [[0, 0, 0], [5, 0, 1]],
    ],
    [[[1 + 2j, 54, 4], [4, 0 + 1j, 45]], [[5, 58, 0], [9, 0, 4]]],
    [[True, False], [True, True], [True, False]],
]


@pytest.mark.parametrize("input", INPUT, ids=str)
def test_argwhere(input):
    anp = np.array(input)
    a = num.array(anp)
    assert np.array_equal(np.argwhere(anp), num.argwhere(a))


@pytest.mark.xfail
def test_argwhere_none():
    # In Numpy, it pass and returns []
    # In cuNumeric, it raises AttributeError:
    # 'NoneType' object has no attribute '_thunk'
    assert np.array_equal(np.argwhere(None), num.argwhere(None))


def test_argwhere_empty():
    anp = np.array([])
    a = num.array(anp)
    assert np.array_equal(np.argwhere(anp), num.argwhere(a))


def test_argwhere_scalar():
    assert np.array_equal(np.argwhere(1), num.argwhere(1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
