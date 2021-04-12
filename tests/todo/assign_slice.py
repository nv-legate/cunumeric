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

import legate.numpy as lg


def test():
    x = lg.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ]
    )
    x[0, :] = 0
    assert lg.array_equal(x[0, :], [0, 0, 0, 0])

    y = lg.array([[20, 30, 40, 50], [70, 80, 90, 100]])
    x[1:4:2] = y
    assert lg.array_equal(x[1, :], [20, 30, 40, 50])
    assert lg.array_equal(x[3, :], [70, 80, 90, 100])

    input_size = 10
    hidden_size = 4
    WLSTM = lg.random.randn(
        input_size + hidden_size + 1, 4 * hidden_size
    ) / lg.sqrt(input_size + hidden_size)
    WLSTM[0, :] = 0  # initialize biases to zero
    assert lg.array_equal(
        WLSTM[0, :],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )

    lg.random.seed(10)
    WLSTM = lg.random.randn(15, 16)
    dIFOGt = lg.random.randn(3, 16)
    dHoutt_in = lg.random.randn(2, 3, 4)
    dHoutt = dHoutt_in.copy()
    dHint = dIFOGt.dot(WLSTM.transpose())
    temp = dHoutt[0, :] + dHint[:, 11:]
    dHoutt[0, :] = temp
    assert not lg.array_equal(dHoutt[0, :], dHoutt_in[0, :])

    return


if __name__ == "__main__":
    test()
