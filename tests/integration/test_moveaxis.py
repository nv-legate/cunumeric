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
from utils.generators import mk_0to1_array

import cunumeric as cn

AXES = [
    (0, 0),
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    ([0, 1], [1, 0]),
]


@pytest.mark.parametrize("ndim", range(2, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("axes", AXES)
def test_moveaxis(ndim, axes):
    source, destination = axes
    np_a = mk_0to1_array(np, (3,) * ndim)
    cn_a = mk_0to1_array(cn, (3,) * ndim)
    np_res = np.moveaxis(np_a, source, destination)
    cn_res = cn.moveaxis(cn_a, source, destination)
    assert np.array_equal(np_res, cn_res)
    # Check that the returned array is a view
    cn_res[:] = 0
    assert cn_a.sum() == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
