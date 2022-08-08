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


def test_basic():
    x = num.array([1, 2, 3])
    assert x[0] == 1
    assert x[1] == 2
    assert x[2] == 3


ARRAYS_4_3_2_1_0 = [
    4 - num.arange(5),
    4 - np.arange(5),
    [4, 3, 2, 1, 0],
]


@pytest.mark.parametrize("arr", ARRAYS_4_3_2_1_0)
def test_scalar_ndarray_as_index(arr):
    offsets = num.arange(5)  # [0, 1, 2, 3, 4]
    offset = offsets[3]  # 3
    assert np.array_equal(arr[offset], 1)
    assert np.array_equal(arr[offset - 2 : offset], [3, 2])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
