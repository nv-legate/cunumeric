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

import cunumeric as num

np.random.seed(0)


def allclose(A, B):
    if B.dtype == np.float32 or B.dtype == np.complex64:
        l2 = (A - B) * np.conj(A - B)
        l2 = np.sqrt(np.sum(l2) / np.sum(A * np.conj(A)))
        return l2 < 1e-6
    else:
        return np.allclose(A, B)


def check_1d_r2c(N, dtype=np.float64):
    print(f"\n=== 1D R2C {dtype}               ===")
    Z = np.random.rand(N).astype(dtype)
    Z_num = num.array(Z)

    out = np.fft.rfft(Z)
    out_num = num.fft.rfft(Z_num)
    assert allclose(out, out_num)
    out = np.fft.rfft(Z, norm="forward")
    out_num = num.fft.rfft(Z_num, norm="forward")
    assert allclose(out, out_num)
    out = np.fft.rfft(Z, n=N // 2)
    out_num = num.fft.rfft(Z_num, n=N // 2)
    assert allclose(out, out_num)
    out = np.fft.rfft(Z, n=N // 2 + 1)
    out_num = num.fft.rfft(Z_num, n=N // 2 + 1)
    assert allclose(out, out_num)
    out = np.fft.rfft(Z, n=N * 2)
    out_num = num.fft.rfft(Z_num, n=N * 2)
    assert allclose(out, out_num)
    out = np.fft.rfft(Z, n=N * 2 + 1)
    out_num = num.fft.rfft(Z_num, n=N * 2 + 1)
    assert allclose(out, out_num)
    # Odd types
    out = np.fft.irfft(Z)
    out_num = num.fft.irfft(Z_num)
    assert allclose(out, out_num)
    out = np.fft.hfft(Z)
    out_num = num.fft.hfft(Z_num)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def check_2d_r2c(N, dtype=np.float64):
    print(f"\n=== 2D R2C {dtype}               ===")
    Z = np.random.rand(*N).astype(dtype)
    Z_num = num.array(Z)

    out = np.fft.rfft2(Z)
    out_num = num.fft.rfft2(Z_num)
    assert allclose(out, out_num)
    out = np.fft.rfft2(Z, norm="forward")
    out_num = num.fft.rfft2(Z_num, norm="forward")
    assert allclose(out, out_num)
    out = np.fft.rfft2(Z, s=(N[0] // 2, N[1] - 2))
    out_num = num.fft.rfft2(Z_num, s=(N[0] // 2, N[1] - 2))
    assert allclose(out, out_num)
    out = np.fft.rfft2(Z, s=(N[0] + 1, N[0] + 2))
    out_num = num.fft.rfft2(Z_num, s=(N[0] + 1, N[0] + 2))
    assert allclose(out, out_num)
    out = np.fft.rfft2(Z, s=(N[0] // 2 + 1, N[0] + 2))
    out_num = num.fft.rfft2(Z_num, s=(N[0] // 2 + 1, N[0] + 2))
    assert allclose(out, out_num)
    out = np.fft.rfft2(Z, axes=[0])
    out_num = num.fft.rfft2(Z_num, axes=[0])
    assert allclose(out, out_num)
    out = np.fft.rfft2(Z, axes=[1])
    out_num = num.fft.rfft2(Z_num, axes=[1])
    assert allclose(out, out_num)
    out = np.fft.rfft2(Z, axes=[-1])
    out_num = num.fft.rfft2(Z_num, axes=[-1])
    assert allclose(out, out_num)
    out = np.fft.rfft2(Z, axes=[-2])
    out_num = num.fft.rfft2(Z_num, axes=[-2])
    assert allclose(out, out_num)
    out = np.fft.rfft2(Z, axes=[0, 1])
    out_num = num.fft.rfft2(Z_num, axes=[0, 1])
    assert allclose(out, out_num)
    out = np.fft.rfft2(Z, axes=[1, 0])
    out_num = num.fft.rfft2(Z_num, axes=[1, 0])
    assert allclose(out, out_num)
    out = np.fft.rfft2(Z, axes=[1, 0, 1])
    out_num = num.fft.rfft2(Z_num, axes=[1, 0, 1])
    assert allclose(out, out_num)
    # Odd types
    out = np.fft.irfft2(Z)
    out_num = num.fft.irfft2(Z_num)
    assert allclose(out, out_num)
    out = np.fft.hfft(Z)
    out_num = num.fft.hfft(Z_num)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def check_3d_r2c(N, dtype=np.float64):
    print(f"\n=== 3D R2C {dtype}               ===")
    Z = np.random.rand(*N).astype(dtype)
    Z_num = num.array(Z)

    out = np.fft.rfftn(Z)
    out_num = num.fft.rfftn(Z_num)
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, norm="forward")
    out_num = num.fft.rfftn(Z_num, norm="forward")
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, norm="ortho")
    out_num = num.fft.rfftn(Z_num, norm="ortho")
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, s=(N[0] - 1, N[1] - 2, N[2] // 2))
    out_num = num.fft.rfftn(Z_num, s=(N[0] - 1, N[1] - 2, N[2] // 2))
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, s=(N[0] + 1, N[1] + 2, N[2] + 3))
    out_num = num.fft.rfftn(Z_num, s=(N[0] + 1, N[1] + 2, N[2] + 3))
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, axes=[0])
    out_num = num.fft.rfftn(Z_num, axes=[0])
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, axes=[1])
    out_num = num.fft.rfftn(Z_num, axes=[1])
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, axes=[2])
    out_num = num.fft.rfftn(Z_num, axes=[2])
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, axes=[-1])
    out_num = num.fft.rfftn(Z_num, axes=[-1])
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, axes=[-2])
    out_num = num.fft.rfftn(Z_num, axes=[-2])
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, axes=[-3])
    out_num = num.fft.rfftn(Z_num, axes=[-3])
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, axes=[2, 1])
    out_num = num.fft.rfftn(Z_num, axes=[2, 1])
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, axes=[0, 2])
    out_num = num.fft.rfftn(Z_num, axes=[0, 2])
    assert allclose(out, out_num)
    out = np.fft.rfftn(Z, axes=[0, 2, 1, 1, -1])
    out_num = num.fft.rfftn(Z_num, axes=[0, 2, 1, 1, -1])
    assert allclose(out, out_num)
    # Odd types
    out = np.fft.fftn(Z)
    out_num = num.fft.fftn(Z_num)
    assert allclose(out, out_num)
    out = np.fft.ifftn(Z)
    out_num = num.fft.ifftn(Z_num)
    assert allclose(out, out_num)
    out = np.fft.irfftn(Z)
    out_num = num.fft.irfftn(Z_num)
    assert allclose(out, out_num)
    out = np.fft.hfft(Z)
    out_num = num.fft.hfft(Z_num)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def test_deferred_1d():
    check_1d_r2c(N=10001)
    check_1d_r2c(N=10001, dtype=np.float32)


def test_deferred_2d():
    check_2d_r2c(N=(128, 512))
    check_2d_r2c(N=(128, 512), dtype=np.float32)


def test_deferred_3d():
    check_3d_r2c(N=(64, 40, 100))
    check_3d_r2c(N=(64, 40, 100), dtype=np.float32)


def test_eager_1d():
    check_1d_r2c(N=153)
    check_1d_r2c(N=153, dtype=np.float32)


def test_eager_2d():
    check_2d_r2c(N=(28, 10))
    check_2d_r2c(N=(28, 10), dtype=np.float32)


def test_eager_3d():
    check_3d_r2c(N=(6, 10, 12))
    check_3d_r2c(N=(6, 10, 12), dtype=np.float32)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
