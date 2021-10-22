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

import cunumeric as lg


def test():
    pythonX = np.reshape(np.linspace(0, 10001, 10000, dtype=int), (100, 100))
    x = lg.array(pythonX)

    pythonY = np.sum(pythonX, axis=0)
    y = lg.sum(x, axis=0)
    assert np.array_equal(pythonY, y)

    pythonY = np.sum(pythonX, axis=1)
    y = lg.sum(x, axis=1)
    assert np.array_equal(pythonY, y)

    return


if __name__ == "__main__":
    test()
