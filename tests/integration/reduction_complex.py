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
    numpyX = np.array([1 + 4j, 2 + 5j, 3 + 6j], np.complex64)
    x = num.array(numpyX)

    z = num.sum(x)
    assert num.all(num.abs(z - np.sum(numpyX)) < 1e-5)

    z = num.prod(x)
    assert num.all(num.abs(z - np.prod(numpyX)) < 1e-5)
    return


if __name__ == "__main__":
    test()
