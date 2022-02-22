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
from test_tools import asserts

import cunumeric as num


def test():
    x = num.ndarray.convert_to_cunumeric_ndarray([])
    y = np.array([])
    asserts.assert_equal(x, y)
    z = num.array([])
    asserts.assert_equal(y, z)

    x = num.ndarray.convert_to_cunumeric_ndarray([[]])
    y = np.array([[]])
    asserts.assert_equal(x, y)
    z = num.array([[]])
    asserts.assert_equal(y, z)

    x = num.ndarray.convert_to_cunumeric_ndarray([[[]]])
    y = np.array([[[]]])
    asserts.assert_equal(x, y)
    z = num.array([[[]]])
    asserts.assert_equal(y, z)

    x = num.ndarray.convert_to_cunumeric_ndarray([[], []])
    y = np.array([[], []])
    asserts.assert_equal(x, y)
    z = num.array([[], []])
    asserts.assert_equal(y, z)

    x = num.ndarray.convert_to_cunumeric_ndarray([1])
    y = np.array([1])
    asserts.assert_equal(x, y)
    z = num.array([1])
    asserts.assert_equal(y, z)

    x = num.ndarray.convert_to_cunumeric_ndarray([[1]])
    y = np.array([[1]])
    asserts.assert_equal(x, y)
    z = num.array([[1]])
    asserts.assert_equal(y, z)

    x = num.ndarray.convert_to_cunumeric_ndarray([[[1]]])
    y = np.array([[[1]]])
    asserts.assert_equal(x, y)
    z = num.array([[[1]]])
    asserts.assert_equal(y, z)

    x = num.ndarray.convert_to_cunumeric_ndarray([[1], [1]])
    y = np.array([[1], [1]])
    asserts.assert_equal(x, y)
    z = num.array([[1], [1]])
    asserts.assert_equal(y, z)

    x = num.ndarray.convert_to_cunumeric_ndarray([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4])
    asserts.assert_equal(x, y)
    z = num.array([1, 2, 3, 4])
    asserts.assert_equal(y, z)


if __name__ == "__main__":
    test()
