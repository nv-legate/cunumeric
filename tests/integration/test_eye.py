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
from utils.utils import check_array_method

import cunumeric as num

N = 5
KS = [0, -1, 1, -2, 2]


@pytest.mark.parametrize("k", KS + [-N, N, -10 * N, 10 * N])
@pytest.mark.parametrize("M", [N, N + 1, N - 1, N * 10, 0])
def test_eye(M, k):
    print_msg = f"np & cunumeric.eye({N},{M}, k={k})"
    check_array_method("eye", [N, M], {"k": k}, print_msg)


@pytest.mark.parametrize("dtype", [np.int32, np.float64, None], ids=str)
@pytest.mark.parametrize("k", KS, ids=str)
def test_square(k, dtype):
    print_msg = f"np & cunumeric.eye({N},k={k},dtype={dtype})"
    check_array_method("eye", [N], {"k": k, "dtype": dtype}, print_msg)


def test_N_zero():
    N = 0
    print_msg = f"np & cunumeric eye({N})"
    check_array_method("eye", [N], {}, print_msg)


def test_M_zero():
    N = 5
    M = 0
    print_msg = f"np & cunumeric eye({N},{M})"
    check_array_method("eye", [N, M], {}, print_msg)


class TestEyeErrors:
    def testBadN(self):
        msg = r"Invalid shape"
        with pytest.raises(ValueError, match=msg):
            num.eye(-1)

        msg = r"expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.eye(5.0)

    def testBadM(self):
        msg = r"Invalid shape"
        with pytest.raises(ValueError, match=msg):
            num.eye(5, -1)

        msg = r"negative dimensions"
        with pytest.raises(ValueError, match=msg):
            num.eye(0, -1)

        msg = r"expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.eye(5, 5.0)

    @pytest.mark.xfail
    def testBadK(self):
        # numpy: raises TypeError
        # cunumeric: the error is found by legate.core, raises struct.error
        with pytest.raises(TypeError):
            num.eye(5, k=0.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
