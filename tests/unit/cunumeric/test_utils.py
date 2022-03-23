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

import cunumeric.utils as m  # module under test
import numpy as np
import pytest

SUPPORTED_DTYPES = [
    np.float16,
    np.float32,
    np.float64,
    float,
    np.int16,
    np.int32,
    np.int64,
    int,
    np.uint16,
    np.uint32,
    np.uint64,
    np.bool_,
    bool,
]


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
    def test_default_top_only(self) -> None:
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


class Test_is_supported_dtype:
    @pytest.mark.parametrize(
        "value", ["foo", 10, 10.2, [], (), {}, set(), None]
    )
    def test_type_bad(self, value) -> None:
        with pytest.raises(AssertionError):  # should be TypeError?
            m.is_supported_dtype(value)

    @pytest.mark.parametrize("value", SUPPORTED_DTYPES)
    def test_supported(self, value) -> None:
        assert m.is_supported_dtype(np.dtype(value))

    # This is just a representative sample, not exhasutive
    @pytest.mark.parametrize(
        "value", [np.float128, np.complex64, np.datetime64]
    )
    def test_unsupported(self, value) -> None:
        assert not m.is_supported_dtype(np.dtype(value))


@pytest.mark.parametrize(
    "shape, volume", [[(), 0], [(10,), 10], [(1, 2, 3), 6]]
)
def test_calculate_volume(shape, volume) -> None:
    assert m.calculate_volume(shape) == volume


def test_get_arg_dtype() -> None:
    dt = m.get_arg_dtype(np.float32)
    assert dt.type is np.void
    assert dt.isalignedstruct
    assert set(dt.fields) == {"arg", "arg_value"}
    assert dt.fields["arg"][0] == np.dtype(np.int64)
    assert dt.fields["arg_value"][0] == np.dtype(np.float32)


def test_get_arg_value_dtype() -> None:
    dt = m.get_arg_dtype(np.float32)
    assert m.get_arg_value_dtype(dt) is np.float32


def test_dot_modes() -> None:
    assert m.dot_modes(0, 0) == ([], [], [])
    assert m.dot_modes(0, 1) == ([], ["A"], ["A"])
    assert m.dot_modes(1, 0) == (["a"], [], ["a"])
    assert m.dot_modes(1, 1) == (["a"], ["a"], [])
    assert m.dot_modes(2, 0) == (["a", "b"], [], ["a", "b"])
    assert m.dot_modes(0, 2) == ([], ["A", "B"], ["A", "B"])
    assert m.dot_modes(2, 1) == (["a", "b"], ["b"], ["a"])
    assert m.dot_modes(1, 2) == (["a"], ["a", "B"], ["B"])
    assert m.dot_modes(2, 2) == (["a", "b"], ["b", "B"], ["a", "B"])


def test_inner_modes() -> None:
    assert m.inner_modes(0, 0) == ([], [], [])
    assert m.inner_modes(0, 1) == ([], ["A"], ["A"])
    assert m.inner_modes(1, 0) == (["a"], [], ["a"])
    assert m.inner_modes(1, 1) == (["a"], ["a"], [])
    assert m.inner_modes(2, 0) == (["a", "b"], [], ["a", "b"])
    assert m.inner_modes(0, 2) == ([], ["A", "B"], ["A", "B"])
    assert m.inner_modes(2, 1) == (["a", "b"], ["b"], ["a"])
    assert m.inner_modes(1, 2) == (["a"], ["A", "a"], ["A"])
    assert m.inner_modes(2, 2) == (["a", "b"], ["A", "b"], ["a", "A"])


@pytest.mark.parametrize("a_ndim, b_ndim", [(0, 0), (0, 1), (1, 0)])
def test_matmul_modes_bad(a_ndim, b_ndim) -> None:
    with pytest.raises(AssertionError):  # should be ValueError?
        m.matmul_modes(a_ndim, b_ndim)


def test_matmul_modes() -> None:
    assert m.matmul_modes(1, 1) == (["z"], ["z"], [])
    assert m.matmul_modes(2, 1) == (["y", "z"], ["z"], ["y"])
    assert m.matmul_modes(1, 2) == (["A"], ["A", "z"], ["z"])
    assert m.matmul_modes(2, 2) == (["y", "A"], ["A", "z"], ["y", "z"])


class Test_tensordot_modes:
    @pytest.mark.parametrize(
        "a_ndim, b_ndim, axes", [(1, 3, 2), (3, 1, 2), (1, 1, 2)]
    )
    def test_bad_single_axis(self, a_ndim, b_ndim, axes) -> None:
        with pytest.raises(ValueError):
            m.tensordot_modes(a_ndim, b_ndim, axes)

    def test_bad_axes_length(self) -> None:
        with pytest.raises(ValueError):
            m.tensordot_modes(1, 3, [(1, 2), (1, 2)])  # len(a_axes) > a_ndim
        with pytest.raises(ValueError):
            m.tensordot_modes(3, 1, [(1, 2), (1, 2)])  # len(b_axes) > b_ndim

    def test_bad_negative_axes(self) -> None:
        with pytest.raises(ValueError):
            m.tensordot_modes(
                3, 2, [(1, -1), (1, 2)]
            )  # any(ax < 0 for ax in a_axes)
        with pytest.raises(ValueError):
            m.tensordot_modes(
                3, 2, [(1, 2), (1, -1)]
            )  # any(ax < 0 for ax in b_axes)

    def test_bad_mismatched_axes(self) -> None:
        with pytest.raises(ValueError):
            m.tensordot_modes(
                4, 4, [(1, 1, 2), (1, 3, 2)]
            )  # len(a_axes) != len(set(a_axes))
        with pytest.raises(ValueError):
            m.tensordot_modes(
                4, 4, [(1, 3, 2), (1, 1, 2)]
            )  # len(b_axes) != len(set(b_axes))

    def test_bad_axes_oob(self) -> None:
        with pytest.raises(ValueError):
            m.tensordot_modes(
                1, 2, [(1, 3), (1, 2)]
            )  # any(ax >= a_ndim for ax in a_axes)
        with pytest.raises(ValueError):
            m.tensordot_modes(
                2, 1, [(1, 2), (1, 3)]
            )  # any(ax >= b_ndim for ax in b_axes)

    def test_single_axis(self):
        assert m.tensordot_modes(0, 0, 0) == ([], [], [])
        assert m.tensordot_modes(2, 2, 1) == (
            ["a", "b"],
            ["b", "B"],
            ["a", "B"],
        )

    def test_tuple_axis(self):
        assert m.tensordot_modes(2, 2, (1, 0)) == (
            ["a", "b"],
            ["b", "B"],
            ["a", "B"],
        )
        assert m.tensordot_modes(2, 2, (0, 1)) == (
            ["a", "b"],
            ["A", "a"],
            ["b", "A"],
        )
        assert m.tensordot_modes(2, 2, (1, 1)) == (
            ["a", "b"],
            ["A", "b"],
            ["a", "A"],
        )

    def test_explicit_axis(self):
        assert m.tensordot_modes(2, 2, ([1], [0])) == (
            ["a", "b"],
            ["b", "B"],
            ["a", "B"],
        )
        assert m.tensordot_modes(2, 2, ([0], [1])) == (
            ["a", "b"],
            ["A", "a"],
            ["b", "A"],
        )
        assert m.tensordot_modes(2, 2, ([1], [1])) == (
            ["a", "b"],
            ["A", "b"],
            ["a", "A"],
        )
        assert m.tensordot_modes(2, 2, ([1, 0], [0, 1])) == (
            ["a", "b"],
            ["b", "a"],
            [],
        )
