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

import cunumeric as num


def allclose(A, B):
    if B.dtype == np.float32 or B.dtype == np.complex64:
        l2 = (A - B) * np.conj(A - B)
        l2 = np.sqrt(np.sum(l2) / np.sum(A * np.conj(A)))
        return l2 < 1e-6
    else:
        return np.allclose(A, B)


def test_1d_hfft(N, dtype=np.float64):
    Z = np.random.rand(N).astype(dtype) + np.random.rand(N).astype(dtype) * 1j
    Z_num = num.array(Z)

    out = np.fft.hfft(Z)
    out_num = num.fft.hfft(Z_num)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def test_1d_hfft_inverse(N, dtype=np.float64):
    Z = np.random.rand(N).astype(dtype)
    Z_num = num.array(Z)

    out = np.fft.ihfft(Z)
    out_num = num.fft.ihfft(Z_num)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


if __name__ == "__main__":
    # Keep errors reproducible
    np.random.seed(0)
    print("DEFERRED")
    print("=== 1D Hermitian double         ===")
    test_1d_hfft(N=10000)
    print("=== 1D Hermitian float          ===")
    test_1d_hfft(N=10000, dtype=np.float32)
    print("=== 1D Hermitian inverse double ===")
    test_1d_hfft_inverse(N=10000)
    print("=== 1D Hermitian inverse float  ===")
    test_1d_hfft_inverse(N=10000, dtype=np.float32)

    print("EAGER")
    print("=== 1D Hermitian double         ===")
    test_1d_hfft(N=110)
    print("=== 1D Hermitian float          ===")
    test_1d_hfft(N=110, dtype=np.float32)
    print("=== 1D Hermitian inverse double ===")
    test_1d_hfft_inverse(N=110)
    print("=== 1D Hermitian inverse float  ===")
    test_1d_hfft_inverse(N=110, dtype=np.float32)
