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

import cunumeric as num


def test_fill_empty_array():
    a_np = np.array([])
    a_num = num.array(a_np)
    a_np.fill(1)
    a_num.fill(1)
    assert np.array_equal(a_np, a_num)


def test_fill_none():
    a_np = np.random.randn(2, 3)
    a_num = num.array(a_np)
    a_np.fill(None)
    a_num.fill(None)
    assert np.array_equal(a_np, a_num, equal_nan=True)


def test_fill_float_to_int():
    a_np = np.full((2, 3), 1)
    a_num = num.array(a_np)
    a_np.fill(0.6)
    a_num.fill(0.6)
    assert np.array_equal(a_np, a_num)


def test_fill_negative_float_to_int():
    a_np = np.full((2, 3), 1)
    a_num = num.array(a_np)
    a_np.fill(-1.3)
    a_num.fill(-1.3)
    assert np.array_equal(a_np, a_num)


def test_fill_int_to_float():
    a_np = np.full((2, 3), 0.5)
    a_num = num.array(a_np)
    a_np.fill(5)
    a_num.fill(5)
    assert np.array_equal(a_np, a_num)


def test_fill_string():
    a_list = ["hello", "hi"]
    a_np = np.array(a_list)
    a_num = num.array(a_np)
    a_np.fill("ok")
    a_num.fill("ok")
    assert np.array_equal(a_np, a_num)


def test_fill_string_to_int():
    a_np = np.full((2, 3), 3)
    a_num = num.array(a_np)
    msg = r"invalid literal for int()"
    with pytest.raises(ValueError, match=msg):
        a_num.fill("OK")


def test_fill_string_to_float():
    a_np = np.random.randn(2, 3)
    a_num = num.array(a_np)
    msg = r"could not convert string to float"
    with pytest.raises(ValueError, match=msg):
        a_num.fill("OK")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
