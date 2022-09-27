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

import cunumeric as cn


@pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM + 1))
def test_diag_indices(ndim):
    np_res = np.diag_indices(10, ndim)
    cn_res = cn.diag_indices(10, ndim)
    assert np.array_equal(np_res, cn_res)


@pytest.mark.parametrize("ndim", range(2, LEGATE_MAX_DIM + 1))
def test_diag_indices_from(ndim):
    shape = (5,) * ndim
    a = np.ones(shape, dtype=int)
    a_cn = cn.array(a)
    np_res = np.diag_indices_from(a)
    cn_res = cn.diag_indices_from(a_cn)
    assert np.array_equal(np_res, cn_res)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
