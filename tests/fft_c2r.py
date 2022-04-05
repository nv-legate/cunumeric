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

import warnings

import numpy as np

import cunumeric as num


def allclose(A, B):
    if B.dtype == np.float32 or B.dtype == np.complex64:
        l2 = (A - B) * np.conj(A - B)
        l2 = np.sqrt(np.sum(l2) / np.sum(A * np.conj(A)))
        return l2 < 1e-6
    else:
        return np.allclose(A, B)


def test_1d_c2r(N, dtype=np.float64):
    Z = np.random.rand(N).astype(dtype) + np.random.rand(N).astype(dtype) * 1j
    Z_num = num.array(Z)

    out = np.fft.irfft(Z)
    out_num = num.fft.irfft(Z_num)
    assert allclose(out, out_num)
    out = np.fft.irfft(Z, norm="forward")
    out_num = num.fft.irfft(Z_num, norm="forward")
    assert allclose(out, out_num)
    out = np.fft.irfft(Z, n=N // 2)
    out_num = num.fft.irfft(Z_num, n=N // 2)
    assert allclose(out, out_num)
    out = np.fft.irfft(Z, n=N // 2 + 1)
    out_num = num.fft.irfft(Z_num, n=N // 2 + 1)
    assert allclose(out, out_num)
    out = np.fft.irfft(Z, n=N * 2)
    out_num = num.fft.irfft(Z_num, n=N * 2)
    assert allclose(out, out_num)
    out = np.fft.irfft(Z, n=N * 2 + 1)
    out_num = num.fft.irfft(Z_num, n=N * 2 + 1)
    assert allclose(out, out_num)
    # Odd types
    out = np.fft.rfft(Z)
    out_num = num.fft.rfft(Z_num)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z)
    out_num = num.fft.ihfft(Z_num)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def test_2d_c2r(N, dtype=np.float64):
    Z = (
        np.random.rand(*N).astype(dtype)
        + np.random.rand(*N).astype(dtype) * 1j
    )
    Z_num = num.array(Z)

    out = np.fft.irfft2(Z)
    out_num = num.fft.irfft2(Z_num)
    assert allclose(out, out_num)
    out = np.fft.irfft2(Z, norm="forward")
    out_num = num.fft.irfft2(Z_num, norm="forward")
    assert allclose(out, out_num)
    out = np.fft.irfft2(Z, s=(N[0] // 2, N[1] - 2))
    out_num = num.fft.irfft2(Z_num, s=(N[0] // 2, N[1] - 2))
    assert allclose(out, out_num)
    out = np.fft.irfft2(Z, s=(N[0] + 1, N[0] + 2))
    out_num = num.fft.irfft2(Z_num, s=(N[0] + 1, N[0] + 2))
    assert allclose(out, out_num)
    out = np.fft.irfft2(Z, s=(N[0] // 2 + 1, N[0] + 2))
    out_num = num.fft.irfft2(Z_num, s=(N[0] // 2 + 1, N[0] + 2))
    assert allclose(out, out_num)
    out = np.fft.irfft2(Z, axes=[0])
    out_num = num.fft.irfft2(Z_num, axes=[0])
    assert allclose(out, out_num)
    out = np.fft.irfft2(Z, axes=[1])
    out_num = num.fft.irfft2(Z_num, axes=[1])
    assert allclose(out, out_num)
    out = np.fft.irfft2(Z, axes=[-1])
    out_num = num.fft.irfft2(Z_num, axes=[-1])
    assert allclose(out, out_num)
    out = np.fft.irfft2(Z, axes=[-2])
    out_num = num.fft.irfft2(Z_num, axes=[-2])
    assert allclose(out, out_num)
    out = np.fft.irfft2(Z, axes=[0, 1])
    out_num = num.fft.irfft2(Z_num, axes=[0, 1])
    assert allclose(out, out_num)
    out = np.fft.irfft2(Z, axes=[1, 0])
    out_num = num.fft.irfft2(Z_num, axes=[1, 0])
    assert allclose(out, out_num)
    out = np.fft.irfft2(Z, axes=[1, 0, 1])
    out_num = num.fft.irfft2(Z_num, axes=[1, 0, 1])
    assert allclose(out, out_num)
    # Odd types
    out = np.fft.rfft2(Z)
    out_num = num.fft.rfft2(Z_num)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z)
    out_num = num.fft.ihfft(Z_num)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def test_3d_c2r(N, dtype=np.float64):
    Z = (
        np.random.rand(*N).astype(dtype)
        + np.random.rand(*N).astype(dtype) * 1j
    )
    Z_num = num.array(Z)

    out = np.fft.irfftn(Z)
    out_num = num.fft.irfftn(Z_num)
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, norm="forward")
    out_num = num.fft.irfftn(Z_num, norm="forward")
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, norm="ortho")
    out_num = num.fft.irfftn(Z_num, norm="ortho")
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, s=(N[0] - 1, N[1] - 2, N[2] // 2))
    out_num = num.fft.irfftn(Z_num, s=(N[0] - 1, N[1] - 2, N[2] // 2))
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, s=(N[0] + 1, N[1] + 2, N[2] + 3))
    out_num = num.fft.irfftn(Z_num, s=(N[0] + 1, N[1] + 2, N[2] + 3))
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, axes=[0])
    out_num = num.fft.irfftn(Z_num, axes=[0])
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, axes=[1])
    out_num = num.fft.irfftn(Z_num, axes=[1])
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, axes=[2])
    out_num = num.fft.irfftn(Z_num, axes=[2])
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, axes=[-1])
    out_num = num.fft.irfftn(Z_num, axes=[-1])
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, axes=[-2])
    out_num = num.fft.irfftn(Z_num, axes=[-2])
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, axes=[-3])
    out_num = num.fft.irfftn(Z_num, axes=[-3])
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, axes=[2, 1])
    out_num = num.fft.irfftn(Z_num, axes=[2, 1])
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, axes=[0, 2])
    out_num = num.fft.irfftn(Z_num, axes=[0, 2])
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z, axes=[0, 2, 1, 1, -1])
    out_num = num.fft.irfftn(Z_num, axes=[0, 2, 1, 1, -1])
    assert allclose(out, out_num)
    # Odd types
    out = np.fft.rfftn(Z)
    out_num = num.fft.rfftn(Z_num)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z)
    out_num = num.fft.ihfft(Z_num)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


if __name__ == "__main__":
    # Keep errors reproducible
    np.random.seed(0)
    print("DEFERRED")
    print("=== 1D C2R double               ===")
    test_1d_c2r(N=10001)
    print("=== 1D C2R float                ===")
    test_1d_c2r(N=10001, dtype=np.float32)
    print("=== 2D C2R double               ===")
    test_2d_c2r(N=(128, 512))
    print("=== 2D C2R float                ===")
    test_2d_c2r(N=(128, 512), dtype=np.float32)
    print("=== 3D C2R double               ===")
    test_3d_c2r(N=(64, 40, 50))
    print("=== 3D C2R float                ===")
    test_3d_c2r(N=(64, 40, 50), dtype=np.float32)

    print("EAGER")
    print("=== 1D C2R double               ===")
    test_1d_c2r(N=78)
    print("=== 1D C2R float                ===")
    test_1d_c2r(N=78, dtype=np.float32)
    print("=== 2D C2R double               ===")
    test_2d_c2r(N=(28, 10))
    print("=== 2D C2R float                ===")
    test_2d_c2r(N=(28, 10), dtype=np.float32)
    print("=== 3D C2R double               ===")
    test_3d_c2r(N=(6, 12, 10))
    print("=== 3D C2R float                ===")
    test_3d_c2r(N=(6, 12, 10), dtype=np.float32)
