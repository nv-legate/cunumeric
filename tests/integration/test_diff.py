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

import cunumeric as cn


@pytest.mark.parametrize(
    "args",
    [
        ((100,), 1, -1, None, None),
        ((100,), 2, -1, None, None),
        ((100,), 3, -1, None, None),
        ((100,), 2, 0, None, None),
        ((10, 10), 2, -1, None, None),
        ((10, 10), 2, 0, None, None),
        ((10, 10), 2, 1, None, None),
        ((100,), 3, -1, [1.0, 2.0], None),
        ((100,), 3, -1, None, [1.0, 2.0]),
        ((100,), 3, -1, [1.0, 2.0], [1.0, 2.0]),
        ((5,), 5, -1, None, None),
        ((5,), 6, 0, None, None),
        ((5, 5), 5, 1, None, None),
        ((5, 5), 6, 1, None, None),
    ],
)
def test_diff(args):
    shape, n, axis, prepend, append = args
    num = np.random.random(shape)
    cun = cn.array(num)

    # We are not adopting the np._NoValue default arguments
    # for this function, as no special behavior is needed on None.
    n_prepend = np._NoValue if prepend is None else prepend
    n_append = np._NoValue if append is None else append
    res_num = np.diff(num, n=n, axis=axis, prepend=n_prepend, append=n_append)
    res_cn = cn.diff(cun, n=n, axis=axis, prepend=prepend, append=append)

    assert allclose(res_num, res_cn)


def test_diff_nzero():
    a = cn.ones(100)
    ad = cn.diff(a, n=0)
    assert a is ad


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
