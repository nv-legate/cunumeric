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
    xn = np.array(
        [[1 + 2j, 3 - 4j, 5 + 6j], [7 - 8j, -9 + 10j, -11 - 12j]], np.complex
    )
    x = lg.array(xn)

    assert lg.all(lg.abs(lg.sin(x) - np.sin(xn)) < 1e-5)
    assert lg.all(lg.abs(lg.cos(x) - np.cos(xn)) < 1e-5)
    assert lg.all(lg.abs(lg.exp(x) - np.exp(xn)) < 1e-5)
    assert lg.all(lg.abs(lg.tanh(x) - np.tanh(xn)) < 1e-5)
    assert lg.all(lg.abs(lg.sqrt(x) - np.sqrt(xn)) < 1e-5)
    return


if __name__ == "__main__":
    test()
