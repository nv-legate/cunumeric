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
from legate.core import LEGATE_MAX_DIM

import cunumeric as num


@pytest.mark.parametrize("write", (None, False, True, 1, -1, 100, "11"))
def test_set_value(write):
    array_np = np.array(3)
    array_num = num.array(3)
    assert array_np.flags == array_num.flags
    array_np.setflags(write)
    array_num.setflags(write)
    assert array_np.flags == array_num.flags


def test_array_default_flags():
    array_np = np.array([0, 0, 0, 0, 0])
    array_num = num.array([0, 0, 0, 0, 0])
    assert array_np.flags["C_CONTIGUOUS"] == array_num.flags["C_CONTIGUOUS"]
    assert array_np.flags["F_CONTIGUOUS"] == array_num.flags["F_CONTIGUOUS"]
    assert array_np.flags["WRITEABLE"] == array_num.flags["WRITEABLE"]
    assert array_np.flags["ALIGNED"] == array_num.flags["ALIGNED"]
    assert (
        array_np.flags["WRITEBACKIFCOPY"] == array_num.flags["WRITEBACKIFCOPY"]
    )
    # array_np.flags
    #          C_CONTIGUOUS : True
    #          F_CONTIGUOUS : True
    #          OWNDATA : True
    #          WRITEABLE : True
    #          ALIGNED : True
    #          WRITEBACKIFCOPY : False
    # array_num.flags
    #          C_CONTIGUOUS : True
    #          F_CONTIGUOUS : True
    #          OWNDATA : False
    #          WRITEABLE : True
    #          ALIGNED : True
    #          WRITEBACKIFCOPY : False


@pytest.mark.xfail
def test_no_writable():
    array_np = np.array([0, 0, 0, 0, 0])
    array_num = num.array([0, 0, 0, 0, 0])
    array_np.setflags(0)
    array_num.setflags(0)
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        array_np[2] = 1
        # Numpy raises ValueError: assignment destination is read-only
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        array_num[2] = 1
        # cuNumeric set value as [0 0 1 0 0]


@pytest.mark.xfail
def test_writeable():
    array_np = np.array([0, 0, 0, 0, 0])
    array_num = num.array([0, 0, 0, 0, 0])
    array_np.setflags(1)
    array_num.setflags(1)
    # cuNumeric raises ValueError: cannot set WRITEABLE flag to
    # True of this array
    array_np[2] = 1
    array_num[2] = 1
    assert array_np.flags == array_num.flags
    assert np.array_equal(array_np, array_num)


def test_logic():
    shape = (3, 3)
    array_np = np.random.randint(1, 100, shape, dtype=int)
    array_num = num.array(array_np)
    array_np.setflags(write=False, align=False)
    array_num.setflags(write=False, align=False)

    expected_exc = ValueError
    with pytest.raises(expected_exc):
        array_np.setflags(uic=True)
        # Numpy raises ValueError: cannot set WRITEBACKIFCOPY flag to True
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        array_num.setflags(uic=True)
        # cuNumeric raises ValueError: cannot set WRITEBACKIFCOPY flag to True


@pytest.mark.xfail
@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_set_write_true(ndim):
    shape = (3,) * ndim
    array_np = np.random.randint(1, 100, shape, dtype=int)
    array_num = num.array(array_np)
    array_np.setflags(write=True)
    array_num.setflags(write=True)
    # cuNumeric raises ValueError: cannot set WRITEABLE flag to True
    # of this array
    assert array_np.flags["WRITEABLE"] == array_num.flags["WRITEABLE"]


@pytest.mark.xfail
@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_set_write_false(ndim):
    shape = (3,) * ndim
    array_np = np.random.randint(1, 100, shape, dtype=int)
    array_num = num.array(array_np)
    array_np.setflags(write=False)
    array_num.setflags(write=False)
    assert array_np.flags["WRITEABLE"] == array_num.flags["WRITEABLE"]


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_set_align_true(ndim):
    shape = (3,) * ndim
    array_np = np.random.randint(1, 100, shape, dtype=int)
    array_num = num.array(array_np)
    array_np.setflags(align=True)
    array_num.setflags(align=True)
    assert array_np.flags["ALIGNED"] == array_num.flags["ALIGNED"]


@pytest.mark.xfail
@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_set_align_false(ndim):
    shape = (3,) * ndim
    array_np = np.random.randint(1, 100, shape, dtype=int)
    array_num = num.array(array_np)
    array_np.setflags(align=False)
    array_num.setflags(align=False)
    assert array_np.flags["ALIGNED"] == array_num.flags["ALIGNED"]


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
