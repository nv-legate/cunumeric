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
    a = lg.random.random((10, 10, 10))
    anp = a.__array__()

    b = lg.flip(a)
    bnp = np.flip(anp)

    assert lg.array_equal(b, bnp)

    b = lg.flip(a, axis=0)
    bnp = np.flip(anp, axis=0)

    assert lg.array_equal(b, bnp)

    b = lg.flip(a, axis=1)
    bnp = np.flip(anp, axis=1)

    assert lg.array_equal(b, bnp)

    b = lg.flip(a, axis=(0, 2))
    bnp = np.flip(anp, axis=(0, 2))

    assert lg.array_equal(b, bnp)


if __name__ == "__main__":
    test()
