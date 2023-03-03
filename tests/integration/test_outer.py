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

import cunumeric as num


def _outer(a_ndim, b_ndim, lib):
    return lib.outer(
        mk_0to1_array(lib, (a_ndim,)), mk_0to1_array(lib, (b_ndim,))
    )


@pytest.mark.parametrize("a_ndim", range(1, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("b_ndim", range(1, LEGATE_MAX_DIM + 1))
def test_basic(a_ndim, b_ndim):
    assert np.array_equal(
        _outer(a_ndim, b_ndim, np), _outer(a_ndim, b_ndim, num)
    )


def test_empty():
    assert np.array_equal(_outer(0, 0, np), _outer(0, 0, num))


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
