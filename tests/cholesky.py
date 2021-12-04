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


def test_diagonal():
    a = num.eye(10) * 10.0
    b = num.linalg.cholesky(a)
    assert num.allclose(b ** 2.0, a)


def test_real(n):
    a = num.random.rand(n, n)
    b = a + a.T + num.eye(n) * n
    c = num.linalg.cholesky(b)
    c_np = np.linalg.cholesky(b.__array__())
    # assert num.allclose(c, c_np)

    np.set_printoptions(precision=3)
    # print(b)
    print(c)
    print(c_np)


def test_complex(n):
    a = num.random.rand(n, n) + num.random.rand(n, n) * 1.0j
    b = a + a.T.conj() + num.eye(n) * n
    c = num.linalg.cholesky(b)
    c_np = np.linalg.cholesky(b.__array__())
    # assert num.allclose(c, c_np)

    np.set_printoptions(precision=3, linewidth=200)
    # print(b)
    print(c)
    print(c_np)


if __name__ == "__main__":
    # test_diagonal()
    test_real(8)
    test_complex(8)
