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


def compare_assert(a_np, a_num):
    if not num.allclose(a_np, a_num):
        print("numpy:")
        print(a_np)
        print("cuNumeric:")
        print(a_num)
        assert False


def test_sort_axis(a_np, a_num, axis):
    compare_assert(a_np, a_num)
    print("Sorting axis " + str(axis) + ":")
    sort_np = np.sort(a_np, axis)
    sort_num = num.sort(a_num, axis)
    compare_assert(sort_np, sort_num)
    argsort_np = np.sort(a_np, axis)
    argsort_num = num.sort(a_num, axis)
    compare_assert(argsort_np, argsort_num)


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
    compare_assert(sortA_np, sortA_num)

    A_num.sort()
    print("Result (inplace): " + str(A_num))
    compare_assert(sortA_np, A_num)

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
    test_sort_axis(A_np, A_num, axis=None)

    return


def test_3D(x_dim, y_dim, z_dim):
    np.random.seed(42)
    A_np = np.array(
        np.random.randint(10, size=x_dim * y_dim * z_dim), dtype=np.int32
    ).reshape(x_dim, y_dim, z_dim)

    A_num = num.array(A_np)
    print("Sorting 3d tensor:\n")
    print(A_np)

    test_sort_axis(A_np, A_num, 2)
    test_sort_axis(A_np, A_num, 1)
    test_sort_axis(A_np, A_num, 0)
    test_sort_axis(A_np, A_num, axis=None)

    return


def test_3D_complex(x_dim, y_dim, z_dim):
    np.random.seed(42)
    A_np = np.array(
        np.random.random(size=x_dim * y_dim * z_dim), dtype=np.complex64
    ).reshape(x_dim, y_dim, z_dim)

    A_num = num.array(A_np)
    print("Sorting 3d tensor:\n")
    print(A_np)

    test_sort_axis(A_np, A_num, 2)
    test_sort_axis(A_np, A_num, 1)
    test_sort_axis(A_np, A_num, 0)
    test_sort_axis(A_np, A_num, axis=None)

    return


def test_custom():
    # 4D still works, >=5D always falls back to numpy
    a = np.arange(4 * 2 * 2 * 4).reshape(4, 2, 2, 4)
    a_num = num.array(a)

    test_sort_axis(a, a_num, 1)
    test_sort_axis(a, a_num, 2)
    test_sort_axis(a, a_num, a.ndim - 1)

    a = np.arange(4 * 4 * 5 * 2 * 3 * 2 * 2 * 2 * 4).reshape(
        4, 4, 5, 2, 3, 2, 2, 2, 4
    )

    a_num = num.array(a)

    test_sort_axis(a, a_num, 1)
    test_sort_axis(a, a_num, 2)
    test_sort_axis(a, a_num, 7)
    test_sort_axis(a, a_num, 4)

    return


def test_api(a=None):
    if a is None:
        a = np.arange(4 * 2 * 3).reshape(4, 2, 3)
    a_num = num.array(a)

    # sort axes
    for i in range(a.ndim):
        compare_assert(np.sort(a, axis=i, kind="stable"), num.sort(a_num, i))

    # flatten
    compare_assert(
        np.sort(a, axis=None, kind="stable"), num.sort(a_num, axis=None)
    )

    # msort
    compare_assert(np.msort(a), num.msort(a_num))

    # sort_complex
    compare_assert(np.sort_complex(a), num.sort_complex(a_num))

    # reverse order sort
    # TODO

    # in-place sort
    copy_a = a.copy()
    copy_a_num = a_num.copy()
    copy_a.sort()
    copy_a_num.sort()
    compare_assert(copy_a, copy_a_num)

    # reverse order sort (in place)
    # TODO

    # argsort
    for i in range(a.ndim):
        compare_assert(a, a_num)
        compare_assert(
            np.argsort(a, axis=i, kind="stable"), num.argsort(a_num, axis=i)
        )

    # flatten
    compare_assert(
        np.argsort(a, axis=None, kind="stable"), num.argsort(a_num, axis=None)
    )

    return


def generate_random(shape, datatype):
    print("Generate random for " + str(datatype))
    a_np = None
    volume = 1
    for i in shape:
        volume *= i

    if np.issubdtype(datatype, np.integer):
        a_np = np.array(
            np.random.randint(
                np.iinfo(datatype).min, np.iinfo(datatype).max, size=volume
            ),
            dtype=datatype,
        )
    elif np.issubdtype(datatype, np.floating):
        a_np = np.array(np.random.random(size=volume), dtype=datatype)
    elif np.issubdtype(datatype, np.complexfloating):
        a_np = np.array(
            np.random.random(size=volume) + np.random.random(size=volume) * 1j,
            dtype=datatype,
        )
    else:
        print("UNKNOWN type " + str(datatype))
        assert False
    return a_np


def test_dtypes():
    np.random.seed(42)
    test_api(generate_random((2, 5, 7), np.uint8))
    test_api(generate_random((8, 5), np.uint16))
    test_api(generate_random((22, 5, 7), np.uint32))

    test_api(generate_random((2, 5, 7), np.int8))
    test_api(generate_random((8, 5), np.int16))
    test_api(generate_random((22, 5, 7), np.int32))
    test_api(generate_random((2, 5, 7), np.int64))

    test_api(generate_random((8, 5), np.float32))
    test_api(generate_random((8, 5), np.float64))
    test_api(generate_random((22, 5, 7), np.double))

    test_api(generate_random((2, 5, 7), np.complex64))
    test_api(generate_random((2, 5, 7), np.complex128))

    return


def test():
    print("\n\n -----------  1D test ---------------\n")
    test_1D()
    print("\n\n -----------  2D test ---------------\n")
    test_2D()
    print("\n\n -----------  3D test (int32) -------\n")
    test_3D(51, 23, 17)
    print("\n\n -----------  3D test (complex) -----\n")
    test_3D_complex(27, 30, 45)
    print("\n\n -----------  4D/5D test-------------\n")
    test_custom()
    print("\n\n -----------  API test --------------\n")
    test_api()
    print("\n\n -----------  dtype test ------------\n")
    test_dtypes()


if __name__ == "__main__":
    test()
