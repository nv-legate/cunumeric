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


def test(ty):
    np.random.seed(42)
    An = np.random.randn(7, 3).astype(ty)
    Bn = np.random.randn(3).astype(ty)
    Cn = An.dot(Bn)

    A = lg.array(An)
    B = lg.array(Bn)
    C = A.dot(B)

    assert np.allclose(C, Cn)

    An = np.random.randn(3).astype(ty)
    Bn = np.random.randn(3, 7).astype(ty)
    Cn = An.dot(Bn)

    A = lg.array(An)
    B = lg.array(Bn)
    C = A.dot(B)

    assert np.allclose(C, Cn)

    An = np.random.randn(3, 7).astype(ty)
    Bn = np.random.randn(3).astype(ty)
    Cn = An.transpose().dot(Bn)

    A = lg.array(An)
    B = lg.array(Bn)
    C = A.transpose().dot(B)

    assert np.allclose(C, Cn)

    An = np.random.randn(3).astype(ty)
    Bn = np.random.randn(7, 3).astype(ty)
    Cn = An.dot(Bn.transpose())

    A = lg.array(An)
    B = lg.array(Bn)
    C = A.dot(B.transpose())

    assert np.allclose(C, Cn)

    A = lg.random.randn(1, 10).astype(ty)
    B = lg.random.randn(10).astype(ty)
    C = A.dot(B)

    An = A.__array__()
    Bn = B.__array__()
    Cn = An.dot(Bn)

    assert np.allclose(C, Cn)

    A = lg.random.randn(10).astype(ty)
    B = lg.random.randn(10, 1).astype(ty)
    C = A.dot(B)

    An = A.__array__()
    Bn = B.__array__()
    Cn = An.dot(Bn)

    assert np.allclose(C, Cn)


if __name__ == "__main__":
    test(np.float16)
    test(np.float64)
    test(np.float32)
