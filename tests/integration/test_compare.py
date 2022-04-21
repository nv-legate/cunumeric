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

import pytest

import cunumeric as np


def test():
    x = np.array([1, 2, 3])
    y = x < 2
    assert np.array_equal(y, [1, 0, 0])
    y = x <= 2
    assert np.array_equal(y, [1, 1, 0])
    y = x > 2
    assert np.array_equal(y, [0, 0, 1])
    y = x >= 2
    assert np.array_equal(y, [0, 1, 1])
    y = x == 2
    assert np.array_equal(y, [0, 1, 0])

    y = (x + 2) * [6, 7, 8]
    assert np.array_equal(y, [18, 28, 40])


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
