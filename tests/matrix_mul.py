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

    np.random.seed(42)
    An = np.random.randn(3, 7)
    Bn = np.random.randn(7, 11)
    Cn = An.dot(Bn)

    A = lg.array(An)
    B = lg.array(Bn)
    C = A.dot(B)

    # print(C)
    # print(Cn)
    assert np.allclose(C, Cn)

    np.random.seed(42)
    An = np.random.randn(3, 7)
    Bn = np.random.randn(11, 7)
    Cn = An.dot(Bn.transpose())

    A = lg.array(An)
    BT = lg.array(Bn)
    C = A.dot(BT.transpose())

    # print(C)
    # print(Cn)
    assert np.allclose(C, Cn)

    np.random.seed(42)
    An = np.random.randn(7, 3)
    Bn = np.random.randn(7, 11)
    Cn = An.transpose().dot(Bn)

    AT = lg.array(An)
    B = lg.array(Bn)
    C = AT.transpose().dot(B)

    # print(C)
    # print(Cn)
    assert np.allclose(C, Cn)

    np.random.seed(42)
    An = np.random.randn(7, 3)
    Bn = np.random.randn(11, 7)
    Cn = An.transpose().dot(Bn.transpose())

    AT = lg.array(An)
    BT = lg.array(Bn)
    C = AT.transpose().dot(BT.transpose())

    # print(C)
    # print(Cn)
    assert np.allclose(C, Cn)

    np.random.seed(42)
    A3np = np.empty((2, 7, 3))
    B3np = np.empty((2, 11, 7))
    A3np[0] = np.random.randn(7, 3)
    B3np[0] = np.random.randn(11, 7)
    Cn = A3np[0].T.dot(B3np[0].T)

    A3 = lg.array(A3np)
    B3 = lg.array(B3np)
    A3[0] = A3np[0]
    B3[0] = B3np[0]
    C = A3[0].T.dot(B3[0].T)

    # print(C)
    # print(Cn)
    assert np.allclose(C, Cn)

    return


if __name__ == "__main__":
    test()
