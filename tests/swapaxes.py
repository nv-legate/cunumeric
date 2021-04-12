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
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a_lg = lg.array(a)
    b_lg = a_lg.swapaxes(0, 1)

    print("small")
    assert lg.array_equal(a_lg.sum(axis=0), b_lg.sum(axis=1))

    a_tall = np.concatenate((a,) * 100)
    a_tall_lg = lg.array(a_tall)
    b_tall_lg = a_tall_lg.swapaxes(0, 1)

    print("tall")
    assert lg.array_equal(a_tall_lg.sum(axis=0), b_tall_lg.sum(axis=1))

    a_wide = np.concatenate((a,) * 100, axis=1)
    a_wide_lg = lg.array(a_wide)
    b_wide_lg = a_wide_lg.swapaxes(0, 1)

    print("wide")
    assert lg.array_equal(a_wide_lg.sum(axis=0), b_wide_lg.sum(axis=1))

    a_big = np.concatenate((a_tall,) * 100, axis=1)
    a_big_lg = lg.array(a_big)
    b_big_lg = a_big_lg.swapaxes(0, 1)

    print("big")
    assert lg.array_equal(a_big_lg.sum(axis=0), b_big_lg.sum(axis=1))


if __name__ == "__main__":
    test()
