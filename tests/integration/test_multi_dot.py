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
from test_tools.generators import mk_0to1_array

import cunumeric as cn

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
    cn_arrays = [mk_0to1_array(cn, shape) for shape in shapes]
    np_res = np.linalg.multi_dot(np_arrays)
    cn_res = cn.linalg.multi_dot(cn_arrays)
    assert np.allclose(np_res, cn_res)

    if len(shapes[0]) == 1:
        if len(shapes[-1]) == 1:
            out = cn.zeros(())
        else:
            out = cn.zeros((shapes[-1][1],))
    else:
        if len(shapes[-1]) == 1:
            out = cn.zeros((shapes[0][0],))
        else:
            out = cn.zeros(
                (
                    shapes[0][0],
                    shapes[-1][1],
                )
            )
    cn_res = cn.linalg.multi_dot(cn_arrays, out=out)
    assert np.allclose(np_res, out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
