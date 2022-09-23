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
from utils.comparisons import allclose
from utils.generators import mk_0to1_array

import cunumeric as cn

# TODO: add negative exponents here, once they become supported
EXPONENTS = [0, 1, 3, 5]


@pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM - 2))
@pytest.mark.parametrize("exp", EXPONENTS)
def test_matrix_power(ndim, exp):
    shape = (3,) * ndim + (2, 2)
    np_a = mk_0to1_array(np, shape)
    cn_a = mk_0to1_array(cn, shape)
    np_res = np.linalg.matrix_power(np_a, exp)
    cn_res = cn.linalg.matrix_power(cn_a, exp)
    assert allclose(np_res, cn_res)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
