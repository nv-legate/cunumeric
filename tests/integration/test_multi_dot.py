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
from utils.comparisons import allclose
from utils.generators import mk_0to1_array

import cunumeric as num

SHAPES = [
    # 2 arrays
    [(6, 5), (5, 4)],
    [(6, 5), (5,)],
    [(5,), (5, 4)],
    [(5,), (5,)],
    # 3 arrays
    [(6, 5), (5, 4), (4, 3)],
    [(6, 5), (5, 4), (4,)],
    [(5,), (5, 4), (4, 3)],
    [(5,), (5, 4), (4,)],
    # 4 arrays
    [(6, 5), (5, 4), (4, 3), (3, 2)],
    [(6, 5), (5, 4), (4, 3), (3,)],
    [(5,), (5, 4), (4, 3), (3, 2)],
    [(5,), (5, 4), (4, 3), (3,)],
]


@pytest.mark.parametrize("shapes", SHAPES)
def test_multi_dot(shapes):
    np_arrays = [mk_0to1_array(np, shape) for shape in shapes]
    num_arrays = [mk_0to1_array(num, shape) for shape in shapes]
    res_np = np.linalg.multi_dot(np_arrays)
    res_num = num.linalg.multi_dot(num_arrays)
    assert allclose(res_np, res_num)

    if len(shapes[0]) == 1:
        if len(shapes[-1]) == 1:
            out = num.zeros(())
        else:
            out = num.zeros((shapes[-1][1],))
    else:
        if len(shapes[-1]) == 1:
            out = num.zeros((shapes[0][0],))
        else:
            out = num.zeros(
                (
                    shapes[0][0],
                    shapes[-1][1],
                )
            )
    res_num = num.linalg.multi_dot(num_arrays, out=out)
    assert allclose(res_np, out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
