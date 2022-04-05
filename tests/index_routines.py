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

import random
from itertools import permutations

import numpy as np
from cunumeric.eager import diagonal_reference
from test_tools.generators import mk_seq_array

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


def advanced_indexing():
    # simple advanced indexing:
    print("advanced indexing test 1")
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    indx = np.array([1, 3, 5])
    res = x[indx]
    x_num = num.array(x)
    indx_num = num.array(indx)
    res_num = x_num[indx_num]
    assert np.array_equal(res, res_num)

    # advanced indexing test when a.ndim ==1 , indx.ndim >1
    print("advanced indexing test 2")
    y = np.array([0, -1, -2, -3, -4, -5])
    y_num = num.array(y)
    index = np.array([[1, 0, 1, 3, 0, 0], [2, 4, 0, 4, 4, 4]])
    index_num = num.array(index)
    assert np.array_equal(y[index], y_num[index_num])

    z = np.array(
        [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
        ]
    )
    z_num = num.array(z)

    # simple 2D case
    print("advanced indexing test 3")
    index_2d = np.array([[1, 2, 0], [5, 5, 5], [2, 3, 4]])
    index_2d_num = num.array(index_2d)
    assert np.array_equal(y[index_2d], y_num[index_2d_num])

    # mismatch dimesion case:
    print("advanced indexing test 4")
    indx = np.array([1, 1])
    indx_num = num.array(indx)
    res = z[indx]
    res_num = z_num[indx_num]
    assert np.array_equal(res, res_num)

    res = z[:, :, indx]
    res_num = z_num[:, :, indx_num]
    assert np.array_equal(res, res_num)

    res = z[:, indx, :]
    res_num = z_num[:, indx_num, :]
    assert np.array_equal(res, res_num)

    # 2d:
    indx = np.array([[1, 1], [1, 0]])
    indx_num = num.array(indx)
    res = z[indx]
    res_num = z_num[indx_num]
    assert np.array_equal(res, res_num)

    res = z[:, indx]
    res_num = z_num[:, indx_num]
    assert np.array_equal(res, res_num)

    # 2 arrays passed to 3d array
    indx0 = np.array([1, 1])
    indx1 = np.array([1, 0])
    indx0_num = num.array(indx0)
    indx1_num = num.array(indx1)
    res = z[indx0, indx1]
    res_num = z_num[indx0_num, indx1_num]
    assert np.array_equal(res, res_num)

    res = z[:, indx0, indx1]
    res_num = z_num[:, indx0_num, indx1_num]
    assert np.array_equal(res, res_num)

    # 2 index arrays passed in a sparse way:
    x = mk_seq_array(np, (3, 4, 5, 6))
    x_num = mk_seq_array(num, (3, 4, 5, 6))
    res = x[:, [0, 1], :, [0, 1]]
    res_num = x_num[:, [0, 1], :, [0, 1]]
    assert np.array_equal(res, res_num)

    res = x[[0, 1], :, [0, 1], 1:]
    res_num = x_num[[0, 1], :, [0, 1], 1:]
    assert np.array_equal(res, res_num)

    res = x[:, [0, 1], :, 1:]
    res_num = x_num[:, [0, 1], :, 1:]
    assert np.array_equal(res, res_num)

    # 2 arrays with broadcasting
    indx0 = np.array([1, 1])
    indx1 = np.array([[1, 0], [1, 0]])
    indx0_num = num.array(indx0)
    indx1_num = num.array(indx1)
    res = z[indx0, indx1]
    res_num = z_num[indx0_num, indx1_num]
    assert np.array_equal(res, res_num)

    # mismatch dimesion case bool:
    indx_bool = np.array([True, False])
    indx_bool_num = num.array(indx_bool)
    res = z[indx_bool]
    res_num = z_num[indx_bool_num]
    assert np.array_equal(res, res_num)

    # test for bool array of the same dimension
    print("advanced indexing test 5")
    index = np.array([True, False, False, True, True, False])
    index_num = num.array(index)
    assert np.array_equal(y[index], y_num[index_num])

    # test in-place assignment fir the case when idx arr
    # is 1d bool array:
    y[index] = 3
    y_num[index_num] = 3
    assert np.array_equal(y, y_num)

    # test for bool array of the same dimension 2D
    print("advanced indexing test 6")
    indx_bool = np.array(
        [
            [
                [False, True, False, False],
                [True, True, False, False],
                [True, False, True, False],
            ],
            [
                [False, True, False, False],
                [True, True, False, False],
                [True, False, True, False],
            ],
        ]
    )
    indx_bool_num = num.array(indx_bool)
    res = z[indx_bool]
    res_num = z_num[indx_bool_num]
    assert np.array_equal(res, res_num)

    # test in-place assignment fir the case when idx arr
    # is 2d bool array:
    z[indx_bool] = 1
    z_num[indx_bool] = 1
    assert np.array_equal(z, z_num)

    # test mixed data
    print("advanced indexing test 7")
    res = z[:, -1]
    res_num = z_num[:, -1]
    assert np.array_equal(res, res_num)

    # case when multiple number of arays is passed
    print("advanced indexing test 8")
    indx0 = np.array([[0, 1], [1, 0], [0, 0]])
    indx1 = np.array([[0, 1], [2, 0], [1, 2]])
    indx2 = np.array([[3, 2], [1, 0], [3, 2]])

    indx0_num = num.array(indx0)
    indx1_num = num.array(indx1)
    indx2_num = num.array(indx2)

    res = z_num[indx0_num, indx1_num, indx2_num]
    res_np = z[indx0, indx1, indx2]
    assert np.array_equal(res, res_np)

    # test in-place assignment fir the case when
    # several index arrays passed
    z_num[indx0_num, indx1_num, indx2_num] = -2
    z[indx0, indx1, indx2] = -2
    assert np.array_equal(z, z_num)

    # indices with broadcast:
    print("advanced indexing test 9")
    indx0 = np.array([[0, 1], [1, 0], [0, 0]])
    indx1 = np.array([[0, 1]])
    indx2 = np.array([[3, 2], [1, 0], [3, 2]])

    indx0_num = num.array(indx0)
    indx1_num = num.array(indx1)
    indx2_num = num.array(indx2)
    res = z_num[indx0_num, indx1_num, indx2_num]
    res_np = z[indx0, indx1, indx2]
    assert np.array_equal(res, res_np)

    # Combining Basic and Advanced Indexing Schemes:
    print("advanced indexing test 10")
    ind0 = np.array([1, 1])
    ind0_num = num.array(ind0)
    res = z[ind0, :, -1]
    res_num = z_num[ind0_num, :, -1]
    assert np.array_equal(res, res_num)

    res = z[ind0, :, [False, True, False, True]]
    res_num = z_num[ind0_num, :, [False, True, False, True]]
    assert np.array_equal(res, res_num)

    res = z[ind0, :, ind0]
    res_num = z_num[ind0_num, :, ind0_num]
    assert np.array_equal(res, res_num)

    res = z[ind0, :, 1:3]
    res_num = z_num[ind0_num, :, 1:3]
    assert np.array_equal(res, res_num)

    res = z[1, :, ind0]
    res_num = z_num[1, :, ind0_num]
    assert np.array_equal(res, res_num)

    x = mk_seq_array(np, (3, 4, 5, 6))
    x_num = mk_seq_array(num, (3, 4, 5, 6))
    res = x[[0, 1], [0, 1], :, 2]
    res_num = x_num[[0, 1], [0, 1], :, 2]
    assert np.array_equal(res, res_num)

    res = x[..., [0, 1], 2]
    res_num = x_num[..., [0, 1], 2]
    assert np.array_equal(res, res_num)

    res = x[:, [0, 1], :, -1]
    res_num = x_num[:, [0, 1], :, -1]
    assert np.array_equal(res, res_num)

    res = x[:, [0, 1], :, 1:]
    res_num = x_num[:, [0, 1], :, 1:]
    assert np.array_equal(res, res_num)

    # In-Place & Augmented Assignments via Advanced Indexing
    # simple 1d case
    y = np.array([0, -1, -2, -3, -4, -5])
    y_num = num.array(y)
    index = np.array([2, 4, 0, 4, 4, 4])
    index_num = num.array(index)
    y[index] = 0
    y_num[index_num] = 0
    assert np.array_equal(y, y_num)

    y[index] = np.array([1, 2, 3, 4, 5, 6])
    y_num[index_num] = num.array([1, 2, 3, 4, 5, 6])
    print(y)
    print(y_num)
    # Order on which data is updated in case when indexing array points to the
    # same daya in the original array is not guaranteed, so we can't call
    # assert np.array_equal(y, y_num) here

    # 2D test
    x = np.array(
        [
            [0.38, -0.16, 0.38, -0.41, -0.04],
            [-0.47, -0.01, -0.18, -0.5, -0.49],
            [0.02, 0.4, 0.33, 0.33, -0.13],
        ]
    )
    indx0 = np.array([0, 1])
    indx1 = np.array([1, 2])
    x_num = num.array(x)
    indx0_num = num.array(indx0)
    indx1_num = num.array(indx1)
    x[indx0, indx1] = 2.0
    x_num[indx0_num, indx1_num] = 2.0
    assert np.array_equal(x, x_num)

    # we do less than LEGATE_MAX_DIM becasue the dimension will be increased by
    # 1 when passig 2d index array
    for ndim in range(2, LEGATE_MAX_DIM):
        a_shape = tuple(random.randint(2, 9) for i in range(ndim))
        np_array = mk_seq_array(np, a_shape)
        num_array = mk_seq_array(num, a_shape)
        # check when N of index arrays == N of dims
        num_tuple_of_indices = tuple()
        np_tuple_of_indices = tuple()
        for i in range(ndim):
            i_shape = (2, 4)
            idx_arr_np = mk_seq_array(np, i_shape) % np_array.shape[i]
            idx_arr_num = num.array(idx_arr_np)
            np_tuple_of_indices += (idx_arr_np,)
            num_tuple_of_indices += (idx_arr_num,)
        assert np.array_equal(
            np_array[np_tuple_of_indices], num_array[num_tuple_of_indices]
        )
        # check when N of index arrays == N of dims
        i_shape = (2, 2)
        idx_arr_np = mk_seq_array(np, i_shape) % np_array.shape[0]
        idx_arr_num = num.array(idx_arr_np)
        assert np.array_equal(np_array[idx_arr_np], num_array[idx_arr_num])
        # test in-place assignment
        np_array[idx_arr_np] = 2
        num_array[idx_arr_num] = 2
        assert np.array_equal(num_array, np_array)
        idx_arr_np = np.array([[1, 0, 1], [1, 1, 0]])
        idx_arr_num = num.array(idx_arr_np)
        assert np.array_equal(
            np_array[:, idx_arr_np], num_array[:, idx_arr_num]
        )
        # test in-place assignment
        np_array[:, idx_arr_np] = 3
        num_array[:, idx_arr_num] = 3
        assert np.array_equal(num_array, np_array)
        if ndim > 2:
            assert np.array_equal(
                np_array[1, :, idx_arr_np], num_array[1, :, idx_arr_num]
            )
            assert np.array_equal(
                np_array[:, idx_arr_np, idx_arr_np],
                num_array[:, idx_arr_num, idx_arr_num],
            )
        if ndim > 3:
            assert np.array_equal(
                np_array[:, idx_arr_np, :, idx_arr_np],
                num_array[:, idx_arr_num, :, idx_arr_num],
            )

    return


def test():
    # --------------------------------------------------------------
    # choose operator
    # --------------------------------------------------------------
    choices1 = [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ]
    a1 = [2, 3, 1, 0]
    num_a1 = num.array(a1)
    num_choices1 = num.array(choices1)

    aout = np.array([2.3, 3.0, 1.2, 0.3])
    num_aout = num.array(aout)

    assert np.array_equal(
        np.choose(a1, choices1, out=aout),
        num.choose(num_a1, num_choices1, out=num_aout),
    )
    assert np.array_equal(aout, num_aout)

    b = [2, 4, 1, 0]
    num_b = num.array(b)
    assert np.array_equal(
        np.choose(b, choices1, mode="clip"),
        num.choose(num_b, num_choices1, mode="clip"),
    )
    assert np.array_equal(
        np.choose(b, choices1, mode="wrap"),
        num.choose(num_b, num_choices1, mode="wrap"),
    )

    a2 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    choices2 = [-10, 10]
    num_a2 = num.array(a2)
    num_choices2 = num.array(choices2)
    assert np.array_equal(
        num.choose(num_a2, num_choices2), np.choose(a2, choices2)
    )

    a3 = np.array([0, 1]).reshape((2, 1, 1))
    c1 = np.array([1, 2, 3]).reshape((1, 3, 1))
    c2 = np.array([-1, -2, -3, -4, -5]).reshape((1, 1, 5))
    num_a3 = num.array(a3)
    num_c1 = num.array(c1)
    num_c2 = num.array(c2)
    assert np.array_equal(
        np.choose(a3, (c1, c2)), num.choose(num_a3, (num_c1, num_c2))
    )

    for ndim in range(1, LEGATE_MAX_DIM + 1):
        tgt_shape = (5,) * ndim
        # try various shapes that broadcast to the target shape
        shapes = [tgt_shape]
        for d in range(len(tgt_shape)):
            sh = list(tgt_shape)
            sh[d] = 1
            shapes.append(tuple(sh))
        for choices_shape in shapes:
            # make sure the choices are between 0 and 1
            np_choices = mk_seq_array(np, choices_shape) % 2
            num_choices = mk_seq_array(num, choices_shape) % 2
            for rhs1_shape in shapes:
                np_rhs1 = np.full(rhs1_shape, 42)
                num_rhs1 = num.full(rhs1_shape, 42)
                for rhs2_shape in shapes:
                    # make sure rhs1 and rhs2 have different values
                    np_rhs2 = np.full(rhs2_shape, 17)
                    num_rhs2 = num.full(rhs2_shape, 17)
                    np_res = np.choose(np_choices, (np_rhs1, np_rhs2))
                    num_res = num.choose(num_choices, (num_rhs1, num_rhs2))
                    assert np.array_equal(np_res, num_res)

    # --------------------------------------------------------------
    # diagonal operator
    # --------------------------------------------------------------
    ad = np.arange(24).reshape(4, 3, 2)
    num_ad = num.array(ad)
    assert np.array_equal(ad.diagonal(), num_ad.diagonal())
    assert np.array_equal(ad.diagonal(0, 1, 2), num_ad.diagonal(0, 1, 2))
    assert np.array_equal(ad.diagonal(1, 0, 2), num_ad.diagonal(1, 0, 2))
    assert np.array_equal(ad.diagonal(-1, 0, 2), num_ad.diagonal(-1, 0, 2))

    # test diagonal
    for ndim in range(2, LEGATE_MAX_DIM + 1):
        a_shape = tuple(random.randint(1, 9) for i in range(ndim))
        np_array = mk_seq_array(np, a_shape)
        num_array = mk_seq_array(num, a_shape)

        # test diagonal
        for axes in permutations(range(ndim), 2):
            diag_size = min(a_shape[axes[0]], a_shape[axes[1]]) - 1
            for offset in range(-diag_size + 1, diag_size):
                assert np.array_equal(
                    np_array.diagonal(offset, axes[0], axes[1]),
                    num_array.diagonal(offset, axes[0], axes[1]),
                )

    # test for diagonal_helper
    for ndim in range(3, LEGATE_MAX_DIM + 1):
        a_shape = tuple(random.randint(1, 9) for i in range(ndim))
        np_array = mk_seq_array(np, a_shape)
        np_array = mk_seq_array(np, a_shape)
        num_array = mk_seq_array(num, a_shape)
        for num_axes in range(3, ndim + 1):
            for axes in permutations(range(ndim), num_axes):
                res_num = num.diagonal(
                    num_array, offset=0, extract=True, axes=axes
                )
                res_ref = diagonal_reference(np_array, axes)
                assert np.array_equal(res_num, res_ref)

    for k in [0, -1, 1, -2, 2]:
        print(f"diag(k={k})")
        a = num.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ]
        )
        an = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ]
        )

        b = num.diag(a, k=k)
        bn = np.diag(an, k=k)
        assert np.array_equal(b, bn)

        c = num.diag(b, k=k)
        cn = np.diag(bn, k=k)
        assert np.array_equal(c, cn)

        d = num.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ]
        )
        dn = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ]
        )

        e = num.diag(d, k=k)
        en = np.diag(dn, k=k)
        assert np.array_equal(e, en)

        f = num.diag(e, k=k)
        fn = np.diag(en, k=k)
        assert np.array_equal(f, fn)

    advanced_indexing()

    return


if __name__ == "__main__":
    test()
