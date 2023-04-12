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

import cunumeric as num

# cunumeric.identity(n, dtype=None, *, like=None)

DTYPE_ALL = [
    "l",
    "L",
    "f",
    "e",
    "d",
    "h",
    "i",
    "H",
    "I",
    "?",
    "b",
    "B",
    "F",
    "D",
]
VALUES = [0, 1, 2, 100]
NEGATIVE_VALUE = [
    None,
    np.inf,
    -np.inf,
    -1,
    -2,
    -3.5,
    3.5,
    5j,
    10 + 20j,
    -100 - 100j,
]


def to_dtype(s):
    return str(np.dtype(s))


@pytest.mark.parametrize("val", VALUES)
def test_value(val):
    res_np = np.identity(val)
    res_num = num.identity(val)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("neg_val", NEGATIVE_VALUE)
def test_value_negative(neg_val):
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.identity(neg_val)
    with pytest.raises(expected_exc):
        num.identity(neg_val)


@pytest.mark.parametrize("dtype", DTYPE_ALL, ids=to_dtype)
def test_dtype(dtype):
    res_np = np.identity(5, dtype=dtype)
    res_num = num.identity(5, dtype=dtype)
    assert np.array_equal(res_np, res_num)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
