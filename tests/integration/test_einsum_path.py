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

import cunumeric as cn

expr = "ij,jk,kl->il"
np_a = np.empty((2, 2))
np_b = np.empty((2, 5))
np_c = np.empty((5, 2))
cn_a = cn.empty((2, 2))
cn_b = cn.empty((2, 5))
cn_c = cn.empty((5, 2))

OPTIMIZE = [
    True,
    False,
    "optimal",
    "greedy",
    ("optimal", 4),
    ["einsum_path", (0, 1), (0, 1)],
]


@pytest.mark.parametrize("optimize", OPTIMIZE)
def test_einsum_path(optimize):
    np_path, _ = np.einsum_path(expr, np_a, np_b, np_c, optimize=optimize)
    cn_path, _ = cn.einsum_path(expr, cn_a, cn_b, cn_c, optimize=optimize)
    assert np_path == cn_path


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
