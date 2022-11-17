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

import cunumeric as num

# TODO: add negative exponents here, once they become supported
EXPONENTS = [0, 1, 3, 5]


@pytest.mark.parametrize("ndim", range(0, LEGATE_MAX_DIM - 2))
@pytest.mark.parametrize("exp", EXPONENTS)
def test_matrix_power(ndim, exp):
    shape = (3,) * ndim + (2, 2)
    a_np = mk_0to1_array(np, shape)
    a_num = mk_0to1_array(num, shape)
    res_np = np.linalg.matrix_power(a_np, exp)
    res_num = num.linalg.matrix_power(a_num, exp)
    assert allclose(res_np, res_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
