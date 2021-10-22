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

import cunumeric as lg


def test():
    x = lg.array([[1, 2], [3, 4], [5, 6]])
    assert lg.array_equal(x[[0, 1, 2], [0, 1, 0]], [1, 4, 5])

    x = lg.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    rows = lg.array([0, 3])
    columns = lg.array([0, 2])
    assert lg.array_equal(x[rows[:, np.newaxis], columns], [[0, 2], [9, 11]])

    zg = lg.array([[-1.2 + 0.5j, 1.2 - 2j], [-2.2 + 3.5j, 4.2 - 6.2j]])
    m = lg.array([[True, False], [False, True]])
    assert lg.array_equal(zg[m], [-1.2 + 0.5j, 4.2 - 6.2j])

    anp = np.array([[[2, 1], [3, 2]], [[2, 4], [4, 1]]])
    a = lg.array(anp)
    nznp = anp < 3
    nzgp = a < 3
    assert lg.array_equal(anp[nznp], a[nzgp])

    y = lg.array(
        [[[True, True], [False, True]], [[True, False], [False, True]]]
    )
    z = lg.nonzero(y)
    assert lg.array_equal(a[z], lg.array([2, 1, 2, 2, 1]))

    np.random.seed(42)
    anp = np.random.randn(10, 10, 4)
    a = lg.array(anp)
    bnp = np.array([3, 4, 6])
    cnp = np.array([1, 4, 5])
    b = lg.array(bnp)
    c = lg.array(cnp)

    assert lg.array_equal(a[b], anp[bnp])
    assert lg.array_equal(a[(b, c)], anp[(b, c)])

    onesnp = np.zeros(10, int)
    ones = lg.zeros(10, int)

    dnp = np.random.randn(20, 4)
    d = lg.array(dnp)
    assert lg.array_equal(dnp[np.where(onesnp)], d[lg.where(ones)])


if __name__ == "__main__":
    test()
