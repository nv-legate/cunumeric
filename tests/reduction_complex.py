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
    numpyX = np.array([1 + 4j, 2 + 5j, 3 + 6j], np.complex64)
    x = lg.array(numpyX)

    z = lg.sum(x)
    assert lg.all(lg.abs(z - np.sum(numpyX)) < 1e-5)

    z = lg.prod(x)
    assert lg.all(lg.abs(z - np.prod(numpyX)) < 1e-5)
    return


if __name__ == "__main__":
    test()
