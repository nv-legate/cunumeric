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
import pytest

import cunumeric as num

rect = num.array([[1, 2, 3], [4, 5, 6]])
square = num.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.mark.parametrize("x", (rect, square), ids=("rect", "square"))
class Test_free_function:
    def test_forward(self, x):
        y = num.transpose(x)
        npx = np.array(x)
        assert num.array_equal(y, np.transpose(npx))

    def test_round_trip(self, x):
        y = num.transpose(x)
        z = num.transpose(y)
        assert num.array_equal(x, z)


@pytest.mark.parametrize("x", (rect, square), ids=("rect", "square"))
class Test_method:
    def test_forward(self, x):
        y = x.transpose()
        npx = np.array(x)
        assert num.array_equal(y, npx.transpose())

    def test_round_trip(self, x):
        y = x.transpose()
        z = y.transpose()
        assert num.array_equal(x, z)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
