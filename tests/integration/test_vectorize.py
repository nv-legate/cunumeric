# Copyright 2023 NVIDIA Corporation
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


def my_func(a, b):
    a = a * 2 + b


def my_func_np(a, b):
    a = a * 2 + b
    return a


def my_func2(A0, B0):
    A0 = A0 * 2 + B0


def my_func_np2(A0, B0):
    A0 = A0 * 2 + B0
    return A0


def test_vectorize():
    func = num.vectorize(my_func)
    a = num.arange(5)
    b = num.ones((5,))
    func(a, b)
    assert np.array_equal(a, [1, 3, 5, 7, 9])

    func = num.vectorize(my_func)
    a = num.arange(5)
    b = 2
    func(a, b)
    assert np.array_equal(a, [2, 4, 6, 8, 10])

    num.vectorize(my_func)
    a = num.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = num.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
    func(a[:2], b[:2])

    a = np.arange(100).reshape((25, 4))
    a_num = num.array(a)

    b = a * 10
    b_num = a_num * 10
    func_np = np.vectorize(my_func_np)
    func_num = num.vectorize(my_func)

    a = func_np(a, b)
    func_num(a_num, b_num)
    assert np.array_equal(a, a_num)

    a[:, 2] = func_np(a[:, 2], b[:, 2])
    func_num(a_num[:, 2], b_num[:, 2])
    assert np.array_equal(a, a_num)

    a[5:10, 2] = func_np(a[5:10, 2], b[1:6, 2])
    func_num(a_num[5:10, 2], b_num[1:6, 2])
    assert np.array_equal(a, a_num)

    a[15:20] = func_np(a[15:20], b[15:20])
    func_num(a_num[15:20], b_num[15:20])
    assert np.array_equal(a, a_num)

    a = np.arange(1000).reshape((25, 10, 4))
    a_num = num.array(a)

    a[:, 2, :] = func_np(a[:, 2, :], 2)
    func_num(a_num[:, 2, :], 2)
    assert np.array_equal(a, a_num)

    a = np.arange(100).reshape((25, 4))
    a_num = num.array(a)

    b = a * 10
    b_num = a_num * 10

    func_np = np.vectorize(my_func_np2)
    func_num = num.vectorize(my_func2)

    a = func_np(a, b)
    func_num(a_num, b_num)
    assert np.array_equal(a, a_num)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
