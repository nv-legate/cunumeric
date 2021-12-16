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


def assert_equal(numarr, nparr):
    for resultnp, resultnum in zip(nparr, numarr):
        assert np.array_equal(resultnp, resultnum)


def test():
    assert num.count_nonzero(num.array([])) == 0
    assert num.count_nonzero(num.array([], dtype="?")) == 0
    assert_equal(num.nonzero([]), np.nonzero([]))

    assert num.count_nonzero(num.array([[], []])) == 0
    assert num.count_nonzero(num.array([[], []], dtype="?")) == 0
    assert_equal(num.nonzero([[], []]), np.nonzero([[], []]))

    assert num.count_nonzero(num.array([0])) == 0
    assert num.count_nonzero(num.array([0], dtype="?")) == 0
    assert_equal(num.nonzero(num.array([0])), ([],))

    assert num.count_nonzero(num.array([1])) == 1
    assert num.count_nonzero(num.array([1], dtype="?")) == 1
    assert_equal(num.nonzero([1]), np.nonzero([1]))

    assert num.count_nonzero(num.array(0)) == 0
    assert num.count_nonzero(num.array(0, dtype="?")) == 0
    assert_equal(num.nonzero(0), np.nonzero(0))

    assert num.count_nonzero(num.array(1)) == 1
    assert num.count_nonzero(num.array(1, dtype="?")) == 1
    assert_equal(num.nonzero(1), np.nonzero(1))

    x = num.array([1, 0, 2, -1, 0, 0, 8])
    x_np = np.array([1, 0, 2, -1, 0, 0, 8])
    assert num.count_nonzero(x) == 4
    assert_equal(num.nonzero(x), np.nonzero(x_np))

    # TBD: this will work once nonzero returns output in row-major output
    # x = num.array([[0, 1, 0], [2, 0, 3]])
    # x_np = np.array([[0, 1, 0], [2, 0, 3]])
    # assert (num.count_nonzero(x) == 3)
    # assert_equal(num.nonzero(x), np.nonzero(x_np))

    x_lg = num.eye(3)
    x_np = np.eye(3)
    assert num.count_nonzero(x_lg) == np.count_nonzero(x_np)
    assert_equal(num.nonzero(x_lg), np.nonzero(x_np))

    # TBD: this will work once nonzero returns output in row-major output
    # x = num.array([[[0, 1], [1, 1], [7, 0], [1, 0], [0, 1]], [[3, 0], [0, 3], [0, 0], [2, 2], [0, 19]]]) # noqa E501
    # x_np = np.array([[[0, 1], [1, 1], [7, 0], [1, 0], [0, 1]], [[3, 0], [0, 3], [0, 0], [2, 2], [0, 19]]]) # noqa E501
    # assert (num.count_nonzero(x) == np.count_nonzero(x_np))
    # assert_equal(num.count_nonzero(x, axis=0), np.count_nonzero(x_np, axis=0)) # noqa E501
    # assert_equal(num.count_nonzero(x, axis=1), np.count_nonzero(x_np, axis=1)) # noqa E501
    # assert_equal(num.count_nonzero(x, axis=2), np.count_nonzero(x_np, axis=2)) # noqa E501
    # assert (num.count_nonzero(x, axis=(0, 1, 2)) == np.count_nonzero(x_np, axis=(0, 1, 2))) # noqa E501
    # assert_equal(num.nonzero(x), np.nonzero(x_np))

    # x_np = np.concatenate((x_np,) * 2000, axis=1)
    # x = num.array(x_np)
    # assert (num.count_nonzero(x) == np.count_nonzero(x_np))
    # assert_equal(num.count_nonzero(x, axis=0), np.count_nonzero(x_np, axis=0)) # noqa E501
    # assert_equal(num.count_nonzero(x, axis=1), np.count_nonzero(x_np, axis=1)) # noqa E501
    # assert_equal(num.count_nonzero(x, axis=2), np.count_nonzero(x_np, axis=2)) # noqa E501
    # assert (num.count_nonzero(x, axis=(0, 1, 2)) == np.count_nonzero(x_np, axis=(0, 1, 2))) # noqa E501
    # assert_equal(num.nonzero(x), np.nonzero(x_np))

    # lg_nonzero = num.nonzero(x)
    # np_nonzero = np.nonzero(x_np)
    # assert_equal(lg_nonzero, np_nonzero)

    assert_equal(num.nonzero(0), ([],))
    assert_equal(num.nonzero(1), ([0],))

    x_np = np.random.randn(100)
    indices = np.random.choice(
        np.arange(x_np.size), replace=False, size=int(x_np.size * 0.2)
    )
    x_np[indices] = 0
    x = num.array(x_np)
    assert num.count_nonzero(x) == np.count_nonzero(x_np)
    lg_nonzero = num.nonzero(x)
    np_nonzero = np.nonzero(x_np)
    assert_equal(lg_nonzero, np_nonzero)

    # x_np = x_np.reshape(10, 10)
    # x = num.array(x_np)
    # assert (num.count_nonzero(x) == np.count_nonzero(x_np))
    # lg_nonzero = num.nonzero(x)
    # np_nonzero = np.nonzero(x_np)
    # assert_equal(lg_nonzero, np_nonzero)

    return


if __name__ == "__main__":
    test()
