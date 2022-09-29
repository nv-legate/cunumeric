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

import numpy as np
import pytest
from legate.core import LEGATE_MAX_DIM
from utils.generators import mk_0to1_array

import cunumeric as cn

AXES = [
    (0, 0),
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    ([0, 1], [1, 0]),
]


@pytest.mark.parametrize("ndim", range(2, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("axes", AXES)
def test_moveaxis(ndim, axes):
    source, destination = axes
    np_a = mk_0to1_array(np, (3,) * ndim)
    cn_a = mk_0to1_array(cn, (3,) * ndim)
    np_res = np.moveaxis(np_a, source, destination)
    cn_res = cn.moveaxis(cn_a, source, destination)
    assert np.array_equal(np_res, cn_res)
    # Check that the returned array is a view
    cn_res[:] = 0
    assert cn_a.sum() == 0


def test_moveaxis_with_empty_axis():
    np_a = np.ones((3, 4, 5))
    cn_a = cn.ones((3, 4, 5))

    axes = ([], [])
    source, destination = axes

    np_res = np.moveaxis(np_a, source, destination)
    cn_res = cn.moveaxis(cn_a, source, destination)
    assert np.array_equal(np_res, cn_res)


EMPTY_ARRAYS = [
    [],
    [[]],
    [[], []],
]


@pytest.mark.parametrize("a", EMPTY_ARRAYS)
def test_moveaxis_with_empty_array(a):
    axes = (0, -1)
    source, destination = axes

    np_res = np.moveaxis(a, source, destination)
    cn_res = cn.moveaxis(a, source, destination)
    assert np.array_equal(np_res, cn_res)


class TestMoveAxisErrors:
    def setup(self):
        self.x = cn.ones((3, 4, 5))

    def test_repeated_axis(self):
        msg = "repeated axis"
        with pytest.raises(ValueError, match=msg):
            cn.moveaxis(self.x, [0, 0], [1, 0])

        with pytest.raises(ValueError, match=msg):
            cn.moveaxis(self.x, [0, 1], [0, -3])

    def test_axis_out_of_bound(self):
        msg = "out of bound"
        with pytest.raises(np.AxisError, match=msg):
            cn.moveaxis(self.x, [0, 3], [0, 1])

        with pytest.raises(np.AxisError, match=msg):
            cn.moveaxis(self.x, [0, 1], [0, -4])

        with pytest.raises(np.AxisError, match=msg):
            cn.moveaxis(self.x, 4, 0)

        with pytest.raises(np.AxisError, match=msg):
            cn.moveaxis(self.x, 0, -4)

    def test_axis_with_different_length(self):
        msg = "arguments must have the same number of elements"
        with pytest.raises(ValueError, match=msg):
            cn.moveaxis(self.x, [0], [1, 0])

    def test_axis_with_bad_type(self):
        msg = "integer argument expected, got float"
        with pytest.raises(TypeError, match=msg):
            cn.moveaxis(self.x, [0.0, 1], [1, 0])

        with pytest.raises(TypeError, match=msg):
            cn.moveaxis(self.x, [0, 1], [1, 0.0])

        msg = "'NoneType' object is not iterable"
        with pytest.raises(TypeError, match=msg):
            cn.moveaxis(self.x, None, 0)

        with pytest.raises(TypeError, match=msg):
            cn.moveaxis(self.x, 0, None)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
