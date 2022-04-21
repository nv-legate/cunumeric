# Copyright 2021-2022 NVIDIA Corporation
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


def test():
    numpyX = np.array([[True, True, False], [True, True, True]])
    x = num.array(numpyX)

    print(num.all(x))
    print(np.all(numpyX))

    print(num.any(x))
    print(np.any(numpyX))

    assert num.array_equal(num.all(x), np.all(numpyX))
    assert num.array_equal(num.all(x, axis=0), np.all(numpyX, axis=0))

    assert num.array_equal(num.any(x), np.any(numpyX))
    assert num.array_equal(num.any(x, axis=0), np.any(numpyX, axis=0))

    arr2 = [5, 10, 0, 100]
    carr2 = num.array(arr2)
    assert num.array_equal(num.all(carr2), np.all(arr2))

    assert num.array_equal(num.all(num.nan), np.all(np.nan))

    arr3 = [[0, 0], [0, 0]]
    carr3 = num.array(arr3)
    print(num.all(carr3))
    print(np.all(arr3))
    assert num.array_equal(num.all(carr3), np.all(arr3))

    #    print (num.all([[True, True], [False, True]], where=[[True],
    #     [False]]))

    #    numpyY = np.array([[True, False], [True, True]])
    #    y = num.array(numpyY)

    #    assert num.array_equal(
    #        num.all(y, where=[True, False]),
    #        np.all(numpyX, where=[True, False])
    #    )
    #    assert num.array_equal(
    #        num.any(y, where=[[True], [False]]),
    #       np.any(numpyX, where=[[True], [False]])
    #    )

    assert num.array_equal(num.all([-1, 4, 5]), np.all([-1, 4, 5]))
    assert num.array_equal(num.any([-1, 4, 5]), np.any([-1, 4, 5]))

    assert num.equal(num.all(num.nan), np.all(np.nan))
    assert num.equal(num.any(num.nan), np.any(np.nan))

    return


if __name__ == "__main__":
    test()
