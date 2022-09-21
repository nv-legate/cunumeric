# Copyright 2022 NVIDIA Corporation
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
from legate.core import LEGATE_MAX_DIM

import cunumeric as num


@pytest.mark.parametrize("ndim", range(LEGATE_MAX_DIM + 1))
def test_ndarray(ndim):
    shape = (4,) * ndim
    a = num.ones(shape)
    a_np = np.array(a)

    assert np.ndim(a_np) == num.ndim(a)


@pytest.mark.parametrize("input", (42, [0, 1, 2], [[0, 1, 2], [3, 4, 5]]))
def test_python_values(input):
    assert np.ndim(input) == num.ndim(input)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
