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
    anp = np.arange(100).reshape(10, 10)
    a = lg.array(anp)

    bnp = np.reshape(anp, (10, 5, 2))
    b = lg.reshape(a, (10, 5, 2))
    assert np.array_equal(bnp, b)

    cnp = np.reshape(anp, (100,))
    c = lg.reshape(a, (100,))
    assert np.array_equal(cnp, c)

    dnp = np.ravel(anp)
    d = lg.ravel(a)
    assert np.array_equal(dnp, d)


if __name__ == "__main__":
    test()
