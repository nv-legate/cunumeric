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

np.random.seed(42)


def test_sum():
    b = np.random.random((10, 12, 13))
    a = num.array(b)
    assert np.allclose(a, b)

    lg_sum = num.sum(a)
    np_sum = np.sum(b)
    assert np.allclose(np_sum, lg_sum)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
