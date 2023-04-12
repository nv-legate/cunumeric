# Copyright 2021-2022 NVIDIA Corporation
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

import cunumeric as num

ARG_FUNCS = ("nanargmax",)


# test apis
class TestNanReduction:
    """
    These are positive cases compared with numpy
    """

    # TODO: randomize location of NaNs and check

    # check without Nans
    @pytest.mark.parametrize("func_name", ARG_FUNCS)
    @pytest.mark.parametrize("ndim", range(LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_simple(self, func_name, ndim, keepdims):
        shape = (5,) * ndim
        in_np = np.random.random(shape)
        in_num = num.array(in_np)

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        assert np.array_equal(
            func_np(in_np, keepdims=keepdims),
            func_num(in_num, keepdims=keepdims),
        )


# test error messages

if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
