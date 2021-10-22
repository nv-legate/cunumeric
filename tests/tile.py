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

import cunumeric as lg


def test():
    a = lg.array([0, 1, 2])

    b = lg.tile(a, 4)
    assert lg.array_equal(b, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    c = lg.tile(a, (3, 4))
    assert lg.array_equal(
        c,
        [
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        ],
    )

    d = lg.tile(a, (3, 1, 4))
    assert lg.array_equal(
        d,
        [
            [[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
            [[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
            [[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
        ],
    )

    e = lg.array([[1, 2], [3, 4]])

    f = lg.tile(e, 2)
    assert lg.array_equal(f, [[1, 2, 1, 2], [3, 4, 3, 4]])

    g = lg.tile(e, (2, 1))
    assert lg.array_equal(g, [[1, 2], [3, 4], [1, 2], [3, 4]])

    return


if __name__ == "__main__":
    test()
