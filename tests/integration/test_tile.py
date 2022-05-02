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

import cunumeric as num


def test_1d():
    a = num.array([0, 1, 2])

    b = num.tile(a, 4)
    assert num.array_equal(b, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    c = num.tile(a, (3, 4))
    assert num.array_equal(
        c,
        [
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        ],
    )

    d = num.tile(a, (3, 1, 4))
    assert num.array_equal(
        d,
        [
            [[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
            [[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
            [[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
        ],
    )


def test_2d():
    e = num.array([[1, 2], [3, 4]])

    f = num.tile(e, 2)
    assert num.array_equal(f, [[1, 2, 1, 2], [3, 4, 3, 4]])

    g = num.tile(e, (2, 1))
    assert num.array_equal(g, [[1, 2], [3, 4], [1, 2], [3, 4]])


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
