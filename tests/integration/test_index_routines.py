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

import random
from itertools import permutations

import numpy as np
import pytest
from legate.core import LEGATE_MAX_DIM
from utils.generators import mk_seq_array

import cunumeric as num
from cunumeric.eager import diagonal_reference


class TestChoose1d:
    choices1 = [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ]
    a1 = [2, 3, 1, 0]
    num_a1 = num.array(a1)
    num_choices1 = num.array(choices1)
    b = [2, 4, 1, 0]
    num_b = num.array(b)

    def test_basic(self):
        assert np.array_equal(
            num.choose(self.num_a1, self.num_choices1),
            np.choose(self.a1, self.choices1),
        )

    def test_out_none(self):
        assert np.array_equal(
            num.choose(self.num_a1, self.num_choices1, out=None),
            np.choose(self.a1, self.choices1, out=None),
        )

    def test_out(self):
        aout = np.array([2.3, 3.0, 1.2, 0.3])
        num_aout = num.array(aout)
        assert np.array_equal(
            np.choose(self.a1, self.choices1, out=aout),
            num.choose(self.num_a1, self.num_choices1, out=num_aout),
        )
        assert np.array_equal(aout, num_aout)

    @pytest.mark.parametrize("mode", ("wrap", "clip"), ids=str)
    def test_mode(self, mode):
        assert np.array_equal(
            np.choose(self.b, self.choices1, mode=mode),
            num.choose(self.num_b, self.num_choices1, mode=mode),
        )


def test_choose_2d():
    a2 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    choices2 = [-10, 10]
    num_a2 = num.array(a2)
    num_choices2 = num.array(choices2)
    assert np.array_equal(
        num.choose(num_a2, num_choices2), np.choose(a2, choices2)
    )

    a3 = np.array([0, 1]).reshape((2, 1, 1))
    c1 = np.array([1, 2, 3]).reshape((1, 3, 1))
    c2 = np.array([-1, -2, -3, -4, -5]).reshape((1, 1, 5))
    num_a3 = num.array(a3)
    num_c1 = num.array(c1)
    num_c2 = num.array(c2)
    assert np.array_equal(
        np.choose(a3, (c1, c2)), num.choose(num_a3, (num_c1, num_c2))
    )


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_choose_target_ndim(ndim):
    tgt_shape = (5,) * ndim
    # try various shapes that broadcast to the target shape
    shapes = [tgt_shape]
    for d in range(len(tgt_shape)):
        sh = list(tgt_shape)
        sh[d] = 1
        shapes.append(tuple(sh))
    for choices_shape in shapes:
        # make sure the choices are between 0 and 1
        np_choices = mk_seq_array(np, choices_shape) % 2
        num_choices = mk_seq_array(num, choices_shape) % 2
        for rhs1_shape in shapes:
            np_rhs1 = np.full(rhs1_shape, 42)
            num_rhs1 = num.full(rhs1_shape, 42)
            for rhs2_shape in shapes:
                # make sure rhs1 and rhs2 have different values
                np_rhs2 = np.full(rhs2_shape, 17)
                num_rhs2 = num.full(rhs2_shape, 17)
                np_res = np.choose(np_choices, (np_rhs1, np_rhs2))
                num_res = num.choose(num_choices, (num_rhs1, num_rhs2))
                assert np.array_equal(np_res, num_res)


SHAPES_A = (
    (2, 4),
    (2, 1),
    (1, 4),
    (1, 1),
    (4,),
    (1,),
    (3, 2, 4),
    (2, 3, 2, 4),
    (1, 3, 1, 1),
)


@pytest.mark.parametrize(
    "shape_a", SHAPES_A, ids=lambda shape_a: f"(shape_a={shape_a})"
)
def test_choose_a_array(shape_a):
    shape_choices = (3, 2, 4)
    np_a = mk_seq_array(np, shape_a) % shape_choices[0]
    num_a = mk_seq_array(num, shape_a) % shape_choices[0]
    np_choices = mk_seq_array(np, shape_choices)
    num_choices = mk_seq_array(num, shape_choices)

    np_res = np.choose(np_a, np_choices)
    num_res = num.choose(num_a, num_choices)
    assert np.array_equal(np_res, num_res)


def test_choose_a_scalar():
    shape_choices = (3, 2, 4)
    a = 1
    np_choices = mk_seq_array(np, shape_choices)
    num_choices = mk_seq_array(num, shape_choices)

    np_res = np.choose(a, np_choices)
    num_res = num.choose(a, num_choices)
    assert np.array_equal(np_res, num_res)


@pytest.mark.parametrize("mode", ("wrap", "clip"), ids=str)
@pytest.mark.parametrize(
    "shape_a", ((3, 2, 4), (4,)), ids=lambda shape_a: f"(shape_a={shape_a})"
)
def test_choose_mode(shape_a, mode):
    shape_choices = (3, 2, 4)
    np_a = mk_seq_array(np, shape_a) - 10
    num_a = mk_seq_array(num, shape_a) - 10
    np_choices = mk_seq_array(np, shape_choices)
    num_choices = mk_seq_array(num, shape_choices)

    np_res = np.choose(np_a, np_choices, mode=mode)
    num_res = num.choose(num_a, num_choices, mode=mode)
    assert np.array_equal(np_res, num_res)


def test_choose_out():
    shape_choices = (3, 2, 4)
    shape_a = (2, 4)
    shape_a_out = (2, 4)
    np_a = mk_seq_array(np, shape_a) % shape_choices[0]
    np_a = np_a.astype(np.int32)
    num_a = mk_seq_array(num, shape_a) % shape_choices[0]
    num_a = num_a.astype(
        np.int32
    )  # cuNumeric would convert np.int32 to default type np.int64
    np_choices = mk_seq_array(np, shape_choices)
    num_choices = mk_seq_array(num, shape_choices)
    np_aout = mk_seq_array(np, shape_a_out) - 10
    num_aout = mk_seq_array(num, shape_a_out) - 10

    np_res = np.choose(np_a, np_choices, out=np_aout)
    num_res = num.choose(num_a, num_choices, out=num_aout)
    assert np.array_equal(np_res, num_res)
    assert np.array_equal(np_aout, num_aout)


@pytest.mark.xfail
def test_choose_mode_none():
    # In Numpy, pass and returns array equals default mode
    # In cuNumeric, raises ValueError: mode=None not understood.
    # Must be 'raise', 'wrap', or 'clip'
    shape_choices = (3, 2, 4)
    shape_a = (2, 4)
    np_a = mk_seq_array(np, shape_a) % shape_choices[0]
    num_a = mk_seq_array(num, shape_a) % shape_choices[0]
    np_choices = mk_seq_array(np, shape_choices)
    num_choices = mk_seq_array(num, shape_choices)

    np_res = np.choose(np_a, np_choices, mode=None)
    num_res = num.choose(num_a, num_choices, mode=None)
    assert np.array_equal(np_res, num_res)


class TestChooseErrors:
    def setup_method(self):
        self.shape_choices = (3, 2, 4)
        self.choices = mk_seq_array(num, self.shape_choices)
        self.shape_a = (2, 4)
        self.a = mk_seq_array(num, self.shape_a) % self.shape_choices[0]

    @pytest.mark.parametrize(
        "value", (-1, 3), ids=lambda value: f"(value={value})"
    )
    def test_a_value_out_of_bound(self, value):
        shape_a = (2, 4)
        a = num.full(shape_a, value)
        msg = "invalid entry in choice array"
        with pytest.raises(ValueError, match=msg):
            num.choose(a, self.choices)

    def test_a_value_float(self):
        shape_a = (2, 4)
        a = num.full(shape_a, 1.0)
        with pytest.raises(TypeError):
            num.choose(a, self.choices)

    @pytest.mark.parametrize(
        "shape_a",
        ((3, 4), (2, 2), (2,), (0,)),
        ids=lambda shape_a: f"(shape_a={shape_a})",
    )
    def test_a_invalid_shape(self, shape_a):
        a = mk_seq_array(num, shape_a) % self.shape_choices[0]
        msg = "shape mismatch"
        with pytest.raises(ValueError, match=msg):
            num.choose(a, self.choices)

    @pytest.mark.xfail
    def test_a_none(self):
        # In Numpy, it raises TypeError
        # In cuNumeric, it raises AttributeError:
        # 'NoneType' object has no attribute 'choose'
        with pytest.raises(TypeError):
            num.choose(None, self.choices)

    def test_empty_choices(self):
        msg = "invalid entry in choice array"
        with pytest.raises(ValueError, match=msg):
            num.choose(self.a, [])

    @pytest.mark.xfail
    def test_choices_none(self):
        # In Numpy, it raises TypeError
        # In cuNumeric, it raises IndexError: tuple index out of range
        with pytest.raises(TypeError):
            num.choose(self.a, None)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            num.choose(self.a, self.choices, mode="InvalidValue")

    def test_out_invalid_shape(self):
        aout = mk_seq_array(num, (1, 4))
        with pytest.raises(ValueError):
            num.choose(self.a, self.choices, out=aout)


def test_diagonal():
    ad = np.arange(24).reshape(4, 3, 2)
    num_ad = num.array(ad)
    assert np.array_equal(ad.diagonal(), num_ad.diagonal())
    assert np.array_equal(ad.diagonal(0, 1, 2), num_ad.diagonal(0, 1, 2))
    assert np.array_equal(ad.diagonal(1, 0, 2), num_ad.diagonal(1, 0, 2))
    assert np.array_equal(ad.diagonal(-1, 0, 2), num_ad.diagonal(-1, 0, 2))

    # test diagonal
    for ndim in range(2, LEGATE_MAX_DIM + 1):
        a_shape = tuple(random.randint(1, 9) for i in range(ndim))
        np_array = mk_seq_array(np, a_shape)
        num_array = mk_seq_array(num, a_shape)

        # test diagonal
        for axes in permutations(range(ndim), 2):
            diag_size = min(a_shape[axes[0]], a_shape[axes[1]]) - 1
            for offset in range(-diag_size + 1, diag_size):
                assert np.array_equal(
                    np_array.diagonal(offset, axes[0], axes[1]),
                    num_array.diagonal(offset, axes[0], axes[1]),
                )

    # test for diagonal_helper
    for ndim in range(3, LEGATE_MAX_DIM + 1):
        a_shape = tuple(random.randint(1, 9) for i in range(ndim))
        np_array = mk_seq_array(np, a_shape)
        num_array = mk_seq_array(num, a_shape)
        for num_axes in range(3, ndim + 1):
            for axes in permutations(range(ndim), num_axes):
                res_num = num.diagonal(
                    num_array, offset=0, extract=True, axes=axes
                )
                res_ref = diagonal_reference(np_array, axes)
                assert np.array_equal(res_num, res_ref)


KS = (0, -1, 1, -2, 2)


@pytest.mark.xfail
@pytest.mark.parametrize("k", KS, ids=lambda k: f"(k={k})")
@pytest.mark.parametrize(
    "shape", ((5, 1), (1, 5)), ids=lambda shape: f"(shape={shape})"
)
def test_diagonal_offset(shape, k):
    # for shape=(5, 1) and k=1, 2,
    # for shape=(1, 5) and k=-1, -2,
    # In cuNumeric,  raise ValueError: 'offset'
    # for diag or diagonal must be in range
    # In Numpy, pass and returns empty array
    a = mk_seq_array(num, shape)
    an = mk_seq_array(np, shape)

    b = num.diagonal(a, k)
    bn = np.diagonal(an, k)
    assert np.array_equal(b, bn)


@pytest.mark.parametrize(
    "shape",
    (pytest.param((3, 0), marks=pytest.mark.xfail), (0, 3)),
    ids=lambda shape: f"(shape={shape})",
)
def test_diagonal_empty_array(shape):
    # for shape=(3, 0) and k=0,
    # In cuNumeric,  raise ValueError: 'offset'
    # for diag or diagonal must be in range
    # In Numpy, pass and returns empty array
    a = mk_seq_array(num, shape)
    an = mk_seq_array(np, shape)

    b = num.diagonal(a)
    bn = np.diagonal(an)
    assert np.array_equal(b, bn)


class TestDiagonalErrors:
    def setup_method(self):
        shape = (3, 4, 5)
        self.a = mk_seq_array(num, shape)

    def test_0d_array(self):
        a = num.array(3)
        with pytest.raises(ValueError):
            num.diagonal(a)

    def test_1d_array(self):
        shape = (3,)
        a = mk_seq_array(num, shape)
        with pytest.raises(ValueError):
            num.diagonal(a)

    @pytest.mark.xfail
    def test_array_none(self):
        # In cuNumeric, it raises AttributeError:
        # 'NoneType' object has no attribute 'diagonal'
        # In Numpy, it raises ValueError:
        # diag requires an array of at least two dimensions.
        with pytest.raises(ValueError):
            num.diagonal(None)

    @pytest.mark.parametrize(
        "axes",
        ((0, 0), pytest.param((0, -3), marks=pytest.mark.xfail)),
        ids=lambda axes: f"(axes={axes})",
    )
    def test_axes_same(self, axes):
        # For axes =  (0, -3),
        # In cuNumeric, it raises ValueError:
        # axes must be the same size as ndim for transpose
        # In Numpy, it raises ValueError: axis1 and axis2 cannot be the same
        axis1, axis2 = axes
        msg = "axes passed to _diag_helper should be all different"
        with pytest.raises(ValueError, match=msg):
            num.diagonal(self.a, 0, axis1, axis2)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "axes", ((0, -4), (3, 0)), ids=lambda axes: f"(axes={axes})"
    )
    def test_axes_out_of_bound(self, axes):
        # In Numpy, it raises numpy.AxisError: is out of bounds
        # In cuNumeric, it raises ValueError:
        # axes must be the same size as ndim for transpose
        axis1, axis2 = axes
        with pytest.raises(np.AxisError):
            num.diagonal(self.a, 0, axis1, axis2)

    @pytest.mark.xfail
    def test_axes_float(self):
        # In Numpy, it raise TypeError
        # In cuNumeric, it raises AssertionError
        with pytest.raises(TypeError):
            num.diagonal(self.a, 0, 0.0, 1)

    @pytest.mark.xfail
    def test_axes_none(self):
        # In Numpy, it raise TypeError
        # In cuNumeric, it raises AssertionError
        with pytest.raises(TypeError):
            num.diagonal(self.a, 0, None, 0)

    @pytest.mark.parametrize(
        "k",
        (pytest.param(0.0, marks=pytest.mark.xfail), -1.5, 1.5),
        ids=lambda k: f"(k={k})",
    )
    def test_k_float(self, k):
        # for k=0.0,
        # In cuNumeric, pass
        # In Numpy, raises TypeError: integer argument expected, got float
        with pytest.raises(TypeError):
            num.diagonal(self.a, k)

    def test_k_none(self):
        with pytest.raises(TypeError):
            num.diagonal(self.a, None)


@pytest.mark.parametrize("k", KS, ids=lambda k: f"(k={k})")
@pytest.mark.parametrize(
    "shape",
    (
        (5,),
        (3, 3),
        pytest.param((5, 1), marks=pytest.mark.xfail),
        pytest.param((1, 5), marks=pytest.mark.xfail),
    ),
    ids=lambda shape: f"(shape={shape})",
)
def test_diag(shape, k):
    # for shape=(5, 1) and k=1, 2,
    # for shape=(1, 5) and k=-1, -2,
    # In cuNumeric,  raise ValueError:
    # 'offset' for diag or diagonal must be in range
    # In Numpy, pass and returns empty array
    a = mk_seq_array(num, shape)
    an = mk_seq_array(np, shape)

    b = num.diag(a, k=k)
    bn = np.diag(an, k=k)
    assert np.array_equal(b, bn)


@pytest.mark.parametrize(
    "shape",
    ((0,), pytest.param((3, 0), marks=pytest.mark.xfail), (0, 3)),
    ids=lambda shape: f"(shape={shape})",
)
def test_diag_empty_array(shape):
    # for shape=(3, 0) and k=0,
    # In cuNumeric,  raise ValueError:
    # 'offset' for diag or diagonal must be in range
    # In Numpy, pass and returns empty array
    a = mk_seq_array(num, shape)
    an = mk_seq_array(np, shape)

    b = num.diag(a)
    bn = np.diag(an)
    assert np.array_equal(b, bn)


class TestDiagErrors:
    def test_0d_array(self):
        a = num.array(3)
        msg = "Input must be 1- or 2-d"
        with pytest.raises(ValueError, match=msg):
            num.diag(a)

    def test_3d_array(self):
        shape = (3, 4, 5)
        a = mk_seq_array(num, shape)
        with pytest.raises(ValueError):
            num.diag(a)

    @pytest.mark.xfail
    def test_array_none(self):
        # In cuNumeric, it raises AttributeError,
        # 'NoneType' object has no attribute 'ndim'
        # In Numpy, it raises ValueError, Input must be 1- or 2-d.
        with pytest.raises(ValueError):
            num.diag(None)

    @pytest.mark.parametrize(
        "k",
        (pytest.param(0.0, marks=pytest.mark.xfail), -1.5, 1.5),
        ids=lambda k: f"(k={k})",
    )
    def test_k_float(self, k):
        # for k=0.0,
        # In cuNumeric, pass
        # In Numpy, raises TypeError: integer argument expected, got float
        shape = (3, 3)
        a = mk_seq_array(num, shape)
        with pytest.raises(TypeError):
            num.diag(a, k=k)

    def test_k_none(self):
        shape = (3, 3)
        a = mk_seq_array(num, shape)
        with pytest.raises(TypeError):
            num.diag(a, k=None)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
