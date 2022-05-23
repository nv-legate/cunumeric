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

import random

import numpy as np
import pytest

import cunumeric as cn
from legate.core import LEGATE_MAX_DIM


@pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM))
def test_indices(ndim):
    dimensions = tuple(random.randint(2, 5) for i in range(ndim))

    np_res = np.indices(dimensions)
    cn_res = cn.indices(dimensions)
    assert np.array_equal(np_res, cn_res)

    np_res = np.indices(dimensions, dtype=float)
    cn_res = cn.indices(dimensions, dtype=float)
    assert np.array_equal(np_res, cn_res)

    np_res = np.indices(dimensions, sparse=True)
    cn_res = cn.indices(dimensions, sparse=True)
    for i in range(len(np_res)):
        assert np.array_equal(np_res[i], cn_res[i])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
