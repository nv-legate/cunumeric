# Copyright 2021 NVIDIA Corporation
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

import cunumeric as num


def test_sort_axis(a_np, a_num, axis):
    assert num.allclose(a_np, a_num)
    print("Sorting axis " + str(axis) + ":")
    sort_np = np.sort(a_np, axis)
    sort_num = num.sort(a_num, axis, kind="merge")
    # print(sort_np)
    # print(sort_num)
    assert num.allclose(sort_np, sort_num)


def test_1D():
    np.random.seed(42)
    A_np = np.array(np.random.randint(10, size=30), dtype=np.int32)

    A_num = num.array(A_np)
    print("Sorting array   : " + str(A_np))

    sortA_np = np.sort(A_np)
    print("Result numpy    : " + str(sortA_np))

    # pdb.set_trace()
    sortA_num = num.sort(A_num)
    print("Result cunumeric: " + str(sortA_num))
    assert num.allclose(sortA_np, sortA_num)

    A_num.sort()
    print("Result (inplace): " + str(A_num))
    assert num.allclose(sortA_np, A_num)

    return


def test_2D():
    np.random.seed(42)
    x_dim = 5
    y_dim = 3
    A_np = np.array(
        np.random.randint(10, size=x_dim * y_dim), dtype=np.int32
    ).reshape(x_dim, y_dim)

    A_num = num.array(A_np)
    print("Sorting matrix:\n")
    print(A_num)

    test_sort_axis(A_np, A_num, 1)
    test_sort_axis(A_np, A_num, 0)

    return


def test_3D():
    np.random.seed(42)
    x_dim = 5
    y_dim = 3
    z_dim = 7
    A_np = np.array(
        np.random.randint(10, size=x_dim * y_dim * z_dim), dtype=np.int32
    ).reshape(x_dim, y_dim, z_dim)

    A_num = num.array(A_np)
    print("Sorting 3d tensor:\n")
    print(A_np)

    test_sort_axis(A_np, A_num, 2)
    test_sort_axis(A_np, A_num, 1)
    test_sort_axis(A_np, A_num, 0)

    return


def test_custom():
    a = np.arange(2 * 4).reshape(2, 4)
    a_transpose = np.transpose(a)

    a_transposed_num = num.array([[0, 4], [1, 5], [2, 6], [3, 7]])
    a_num = num.array(a)
    a_num_transposed = a_num.swapaxes(0, 1)

    test_sort_axis(a, a_num, 1)
    test_sort_axis(a_transpose, a_transposed_num, 1)
    test_sort_axis(a_transpose, a_num_transposed, 1)
    test_sort_axis(a_transpose, a_num_transposed, 0)

    return


def test():
    print("\n\n -----------  Custom test ---------------\n")
    test_custom()
    print("\n\n -----------  2D test ---------------\n")
    test_2D()
    print("\n\n -----------  3D test ---------------\n")
    test_3D()
    print("\n\n -----------  1D test ---------------\n")
    test_1D()


if __name__ == "__main__":
    test()
