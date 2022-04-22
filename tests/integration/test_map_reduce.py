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
    x = [1.0, 2, 3]
    y = [4, 5, 6]
    z = x + y
    numpyResult = np.sum(z)
    # print(numpyResult)

    gx = num.array(x)
    gy = num.array(y)
    z = gx + gy
    legate_oldResult = num.sum(z)
    # print(legate_oldResult)

    assert legate_oldResult == numpyResult


if __name__ == "__main__":
    test()
