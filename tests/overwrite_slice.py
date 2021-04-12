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

import legate.numpy as lg


def test():
    x = lg.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x[0:5] = lg.array([11, 12, 13, 14, 15])
    x[5:10] = lg.array([16, 17, 18, 19, 20])
    x[4:8] = lg.array([21, 22, 23, 24])
    assert np.array_equal(x[5:10], [22, 23, 24, 19, 20])
    assert np.array_equal(x, [11, 12, 13, 14, 21, 22, 23, 24, 19, 20])

    anp = np.zeros((5, 6))
    bnp = np.random.random((5, 4))
    cnp = np.random.random((5, 2))
    a = lg.zeros((5, 6))
    b = lg.array(bnp)
    c = lg.array(cnp)
    a[:, :4] = b
    a[:, 0] = 1
    a[:, 3:5] = c
    anp[:, :4] = bnp
    anp[:, 0] = 1
    anp[:, 3:5] = cnp
    assert np.array_equal(a, anp)

    dnp = np.random.random((2, 3, 4))
    enp = np.random.random((2, 3, 4))
    fnp = np.random.random((3, 2))
    d = lg.array(dnp)
    e = lg.array(enp)
    f = lg.array(fnp)
    d[1, :, 0] = 1
    d[1, :, 1:3] = f
    d[0] = e[1]
    dnp[1, :, 0] = 1
    dnp[1, :, 1:3] = fnp
    dnp[0] = enp[1]
    assert np.array_equal(d, dnp)

    return


if __name__ == "__main__":
    test()
