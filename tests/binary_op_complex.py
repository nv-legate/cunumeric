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

from __future__ import division

import numpy as np

import legate.numpy as lg


def test():
    numpyX = np.array([1 + 4j, 2 + 5j, 3 + 6j], np.complex64)
    numpyY = np.array([4 + 7j, 5 + 8j, 6 + 9j], np.complex64)

    x = lg.array(numpyX)
    y = lg.array(numpyY)

    z = x + y
    # print(z)
    assert lg.all(lg.abs(z - (numpyX + numpyY)) < 1e-5)

    z = x + 2
    # print(z)
    assert lg.all(lg.abs(z - (numpyX + 2)) < 1e-5)

    z = 2 + x
    # # print(z)
    assert lg.all(lg.abs(z - (2 + numpyX)) < 1e-5)

    z = x - y
    # # print(z)
    assert lg.all(lg.abs(z - (numpyX - numpyY)) < 1e-5)

    z = x - 2
    # print(z)
    assert lg.all(lg.abs(z - (numpyX - 2)) < 1e-5)

    z = 2 - x
    # # print(z)
    assert lg.all(lg.abs(z - (2 - numpyX)) < 1e-5)

    z = x / y
    # print(z)
    assert lg.all(lg.abs(z - (numpyX / numpyY)) < 1e-5)

    z = x / 2
    # print(z)
    assert lg.all(lg.abs(z - (numpyX / 2)) < 1e-5)

    z = 2 / x
    # print(z)
    assert lg.all(lg.abs(z - (2 / numpyX)) < 1e-5)

    z = x * y
    # print(z)
    assert lg.all(lg.abs(z - (numpyX * numpyY)) < 1e-5)

    z = x * 2
    # print(z)
    assert lg.all(lg.abs(z - (numpyX * 2)) < 1e-5)

    z = 2 * x
    # print(z)
    assert lg.all(lg.abs(z - (2 * numpyX)) < 1e-5)

    z = x ** 5
    # Thrust power computation is not very precise, so 1e-1
    assert lg.all(lg.abs(z - numpyX ** 5) < 1e-1)

    z = 5 ** x
    assert lg.all(lg.abs(z - 5 ** numpyX) < 1e-5)

    z = x ** y
    assert lg.all(lg.abs(z - numpyX ** numpyY) < 1e-5)

    return


if __name__ == "__main__":
    test()
