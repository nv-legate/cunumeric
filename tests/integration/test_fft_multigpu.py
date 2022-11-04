# Copyright 2021 NVIDIA Corporation
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
from utils.comparisons import allclose as _allclose

import cunumeric as num

np.random.seed(0)


def allclose(A, B):
    if B.dtype == np.float32 or B.dtype == np.complex64:
        l2 = (A - B) * np.conj(A - B)
        l2 = np.sqrt(np.sum(l2) / np.sum(A * np.conj(A)))
        return l2 < 1e-6
    else:
        return _allclose(A, B)


def check_1d_c2c(N, dtype=np.float64):
    Z = np.random.rand(N).astype(dtype) + np.random.rand(N).astype(dtype) * 1j
    Z_num = num.array(Z)

    all_kwargs = ({},)

    # print(f"Z={Z}")
    # print(f"Z_num={Z_num}")

    for kwargs in all_kwargs:
        print(f"=== 1D C2C {dtype}, args: {kwargs} ===")
        out = np.fft.fft(Z, **kwargs)
        out_num = num.fft.fft(Z_num, **kwargs)

        # print(f"out={out}")
        # print(f"out_num={out_num}")

        assert allclose(out, out_num)
        out = np.fft.ifft(Z, **kwargs)
        out_num = num.fft.ifft(Z_num, **kwargs)
        assert allclose(out, out_num)


def test_1d():
    check_1d_c2c(N=256)
    check_1d_c2c(N=256, dtype=np.float32)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
