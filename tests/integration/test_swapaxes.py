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

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_small():
    a_num = num.array(a)
    b_num = a_num.swapaxes(0, 1)

    assert num.array_equal(a_num.sum(axis=0), b_num.sum(axis=1))


def test_tall():
    a_tall = np.concatenate((a,) * 100)
    a_tall_num = num.array(a_tall)
    b_tall_num = a_tall_num.swapaxes(0, 1)

    assert num.array_equal(a_tall_num.sum(axis=0), b_tall_num.sum(axis=1))


def test_wide():
    a_wide = np.concatenate((a,) * 100, axis=1)
    a_wide_num = num.array(a_wide)
    b_wide_num = a_wide_num.swapaxes(0, 1)

    assert num.array_equal(a_wide_num.sum(axis=0), b_wide_num.sum(axis=1))


def test_big():
    a_tall = np.concatenate((a,) * 100)
    a_big = np.concatenate((a_tall,) * 100, axis=1)
    a_big_num = num.array(a_big)
    b_big_num = a_big_num.swapaxes(0, 1)

    assert num.array_equal(a_big_num.sum(axis=0), b_big_num.sum(axis=1))


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
