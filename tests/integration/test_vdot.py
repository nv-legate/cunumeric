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

DTYPES = [np.float32, np.complex64]


def _vdot(a_dtype, b_dtype, lib):
    return lib.vdot(
        mk_0to1_array(lib, (5,), dtype=a_dtype),
        mk_0to1_array(lib, (5,), dtype=b_dtype),
    )


@pytest.mark.parametrize("a_dtype", DTYPES)
@pytest.mark.parametrize("b_dtype", DTYPES)
def test(a_dtype, b_dtype):
    assert np.allclose(
        _vdot(a_dtype, b_dtype, np), _vdot(a_dtype, b_dtype, cn)
    )


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
