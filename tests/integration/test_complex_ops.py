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

import cunumeric as num

DTYPES = [np.complex64, np.complex128]


@pytest.mark.parametrize("dtype", DTYPES)
def test_array(dtype):
    x_np = np.array([1 + 4j, 2 + 5j, 3 + 6j], dtype)
    x_num = num.array(x_np)

    assert num.array_equal(x_np.conj(), x_num.conj())
    assert num.array_equal(x_np.real, x_num.real)
    assert num.array_equal(x_np.imag, x_num.imag)


@pytest.mark.parametrize("dtype", DTYPES)
def test_single(dtype):
    x_np = np.array([3 + 6j], dtype)
    x_num = num.array(x_np)

    assert num.array_equal(x_np.conj(), x_num.conj())
    assert num.array_equal(x_np.real, x_num.real)
    assert num.array_equal(x_np.imag, x_num.imag)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
