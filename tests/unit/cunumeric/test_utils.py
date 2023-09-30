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

import inspect
from typing import List, Tuple, Union

import numpy as np
import pytest

import cunumeric.utils as m  # module under test

EXPECTED_SUPPORTED_DTYPES = set(
    [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]
)


class Test_is_advanced_indexing:
    def test_Ellipsis(self):
        assert not m.is_advanced_indexing(...)

    def test_None(self):
        assert not m.is_advanced_indexing(None)

    @pytest.mark.parametrize("typ", EXPECTED_SUPPORTED_DTYPES)
    def test_np_scalar(self, typ):
        assert not m.is_advanced_indexing(typ(10))

    def test_slice(self):
        assert not m.is_advanced_indexing(slice(None, 10))
        assert not m.is_advanced_indexing(slice(1, 10))
        assert not m.is_advanced_indexing(slice(None, 10, 2))

    def test_tuple_False(self):
        assert not m.is_advanced_indexing((..., None, np.int32()))

    def test_tuple_True(self):
        assert m.is_advanced_indexing(([1, 2, 3], np.array([1, 2])))

    def test_advanced(self):
        assert m.is_advanced_indexing([1, 2, 3])
        assert m.is_advanced_indexing(np.array([1, 2, 3]))


def test_find_last_user_stacklevel() -> None:
    n = m.find_last_user_stacklevel()
    assert isinstance(n, int)
    assert n == 1


def test_get_line_number_from_frame() -> None:
    frame = inspect.currentframe()
    result = m.get_line_number_from_frame(frame)
    assert isinstance(result, str)
    filename, lineno = result.split(":")

    # NOTE: this will break if this test filename is changed
    assert filename.endswith("test_utils.py")

    # it would be too fragile to compare more specific than this
    assert int(lineno) > 0


class Test_find_last_user_frames:
    def check_default_top_only(self) -> None:
        result = m.find_last_user_frames(top_only=True)
        assert isinstance(result, str)
        assert "|" not in result
        assert "\n" not in result
        assert len(result.split(":")) == 2

    def test_top_only_True(self) -> None:
        result = m.find_last_user_frames(top_only=True)
        assert isinstance(result, str)
        assert "|" not in result
        assert "\n" not in result
        assert len(result.split(":")) == 2

    def test_top_only_False(self) -> None:
        result = m.find_last_user_frames(top_only=False)
        assert isinstance(result, str)
        assert "|" in result

        # it would be too fragile to compare more specific than this
        assert len(result.split("|")) > 1
        assert all(len(x.split(":")) == 2 for x in result.split("|"))


def test__SUPPORTED_DTYPES():
    assert set(m.SUPPORTED_DTYPES.keys()) == set(
        np.dtype(ty) for ty in EXPECTED_SUPPORTED_DTYPES
    )


class Test_is_supported_dtype:
    @pytest.mark.parametrize("value", ["foo", 10, 10.2, (), set()])
    def test_type_bad(self, value) -> None:
        with pytest.raises(TypeError):
            m.to_core_dtype(value)

    @pytest.mark.parametrize("value", EXPECTED_SUPPORTED_DTYPES)
    def test_supported(self, value) -> None:
        m.to_core_dtype(value)

    # This is just a representative sample, not exhasutive
    @pytest.mark.parametrize("value", [np.float128, np.datetime64, [], {}])
    def test_unsupported(self, value) -> None:
        with pytest.raises(TypeError):
            m.to_core_dtype(value)


@pytest.mark.parametrize(
    "shape, volume", [[(), 0], [(10,), 10], [(1, 2, 3), 6]]
)
def test_calculate_volume(shape, volume) -> None:
    assert m.calculate_volume(shape) == volume


def _dot_modes_oracle(a_ndim: int, b_ndim: int) -> bool:
    a_modes, b_modes, out_modes = m.dot_modes(a_ndim, b_ndim)
    expr = f"{''.join(a_modes)},{''.join(b_modes)}->{''.join(out_modes)}"
    a = np.random.randint(100, size=3**a_ndim).reshape((3,) * a_ndim)
    b = np.random.randint(100, size=3**b_ndim).reshape((3,) * b_ndim)
    return np.array_equal(np.einsum(expr, a, b), np.dot(a, b))


@pytest.mark.parametrize(
    "a, b",
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (0, 2),
        (2, 1),
        (1, 2),
        (2, 2),
        (5, 1),
        (1, 5),
    ],
)
def test_dot_modes(a: int, b: int) -> None:
    assert _dot_modes_oracle(a, b)


def _inner_modes_oracle(a_ndim: int, b_ndim: int) -> bool:
    a_modes, b_modes, out_modes = m.inner_modes(a_ndim, b_ndim)
    expr = f"{''.join(a_modes)},{''.join(b_modes)}->{''.join(out_modes)}"
    a = np.random.randint(100, size=3**a_ndim).reshape((3,) * a_ndim)
    b = np.random.randint(100, size=3**b_ndim).reshape((3,) * b_ndim)
    return np.array_equal(np.einsum(expr, a, b), np.inner(a, b))


@pytest.mark.parametrize(
    "a, b",
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (0, 2),
        (2, 1),
        (1, 2),
        (2, 2),
        (5, 1),
        (1, 5),
    ],
)
def test_inner_modes(a: int, b: int) -> None:
    assert _inner_modes_oracle(a, b)


@pytest.mark.parametrize("a, b", [(0, 0), (0, 1), (1, 0)])
def test_matmul_modes_bad(a: int, b: int) -> None:
    with pytest.raises(ValueError):
        m.matmul_modes(a, b)


def _matmul_modes_oracle(a_ndim: int, b_ndim: int) -> bool:
    a_modes, b_modes, out_modes = m.matmul_modes(a_ndim, b_ndim)
    expr = f"{''.join(a_modes)},{''.join(b_modes)}->{''.join(out_modes)}"
    a = np.random.randint(100, size=3**a_ndim).reshape((3,) * a_ndim)
    b = np.random.randint(100, size=3**b_ndim).reshape((3,) * b_ndim)
    return np.array_equal(np.einsum(expr, a, b), np.matmul(a, b))


@pytest.mark.parametrize(
    "a, b", [(1, 1), (2, 1), (1, 2), (2, 2), (5, 1), (1, 5)]
)
def test_matmul_modes(a: int, b: int) -> None:
    assert _matmul_modes_oracle(a, b)


AxesType = Union[int, Tuple[int, int], Tuple[List[int], List[int]]]


def _tensordot_modes_oracle(a_ndim: int, b_ndim: int, axes: AxesType) -> bool:
    a_modes, b_modes, out_modes = m.tensordot_modes(a_ndim, b_ndim, axes)
    expr = f"{''.join(a_modes)},{''.join(b_modes)}->{''.join(out_modes)}"
    a = np.random.randint(100, size=3**a_ndim).reshape((3,) * a_ndim)
    b = np.random.randint(100, size=3**b_ndim).reshape((3,) * b_ndim)
    return np.array_equal(np.einsum(expr, a, b), np.tensordot(a, b, axes))


class Test_tensordot_modes:
    @pytest.mark.parametrize(
        "a_ndim, b_ndim, axes", [(1, 3, 2), (3, 1, 2), (1, 1, 2)]
    )
    def test_bad_single_axis(self, a_ndim, b_ndim, axes) -> None:
        with pytest.raises(ValueError):
            m.tensordot_modes(a_ndim, b_ndim, axes)

    def test_bad_axes_length(self) -> None:
        with pytest.raises(ValueError):
            # len(a_axes) > a_ndim
            m.tensordot_modes(1, 3, [(1, 2), (1, 2)])

        with pytest.raises(ValueError):
            # len(b_axes) > b_ndim
            m.tensordot_modes(3, 1, [(1, 2), (1, 2)])

        with pytest.raises(ValueError):
            # len(a_axes) != len(b_axes)
            m.tensordot_modes(2, 3, ([0], [0, 1]))

    def test_bad_negative_axes(self) -> None:
        with pytest.raises(ValueError):
            # any(ax < 0 for ax in a_axes)
            m.tensordot_modes(3, 2, [(1, -1), (1, 2)])

        with pytest.raises(ValueError):
            # any(ax < 0 for ax in b_axes)
            m.tensordot_modes(3, 2, [(1, 2), (1, -1)])

    def test_bad_mismatched_axes(self) -> None:
        with pytest.raises(ValueError):
            # len(a_axes) != len(set(a_axes))
            m.tensordot_modes(4, 4, [(1, 1, 2), (1, 3, 2)])

        with pytest.raises(ValueError):
            # len(b_axes) != len(set(b_axes))
            m.tensordot_modes(4, 4, [(1, 3, 2), (1, 1, 2)])

    def test_bad_axes_oob(self) -> None:
        with pytest.raises(ValueError):
            # any(ax >= a_ndim for ax in a_axes)
            m.tensordot_modes(1, 2, [(1, 3), (1, 2)])

        with pytest.raises(ValueError):
            # any(ax >= b_ndim for ax in b_axes)
            m.tensordot_modes(2, 1, [(1, 2), (1, 3)])

    @pytest.mark.parametrize("a, b, axes", [(0, 0, 0), (2, 2, 1)])
    def test_single_axis(self, a: int, b: int, axes: AxesType):
        assert _tensordot_modes_oracle(a, b, axes)

    @pytest.mark.parametrize(
        "a, b, axes",
        [(2, 2, (1, 0)), (2, 2, (0, 1)), (2, 2, (1, 1)), (2, 2, (-1, 0))],
    )
    def test_tuple_axis(self, a: int, b: int, axes: AxesType):
        assert _tensordot_modes_oracle(a, b, axes)

    @pytest.mark.parametrize(
        "a, b, axes",
        [
            (2, 2, ([1], [0])),
            (2, 2, ([0], [1])),
            (2, 2, ([1], [1])),
            (2, 2, ([1, 0], [0, 1])),
        ],
    )
    def test_explicit_axis(self, a: int, b: int, axes: AxesType):
        assert _tensordot_modes_oracle(a, b, axes)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
