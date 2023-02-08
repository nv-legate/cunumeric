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


def allclose(A, B):
    if B.dtype == np.float32 or B.dtype == np.complex64:
        l2 = (A - B) * np.conj(A - B)
        l2 = np.sqrt(np.sum(l2) / np.sum(A * np.conj(A)))
        return l2 < 1e-6
    else:
        return _allclose(A, B)


def check_1d_r2c(N, dtype=np.float64):
    Z = np.random.rand(N).astype(dtype)
    Z_num = num.array(Z)

    all_kwargs = (
        {},
        {"norm": "forward"},
        {"n": N // 2},
        {"n": N // 2 + 1},
        {"n": N * 2},
        {"n": N * 2 + 1},
    )

    for kwargs in all_kwargs:
        print(f"=== 1D R2C {dtype}, args: {kwargs} ===")
        out = np.fft.rfft(Z, **kwargs)
        out_num = num.fft.rfft(Z_num, **kwargs)
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
    Z = np.random.rand(*N).astype(dtype)
    Z_num = num.array(Z)

    all_kwargs = (
        {},
        {"norm": "forward"},
        {"s": (N[0] // 2, N[1] - 2)},
        {"s": (N[0] + 1, N[0] + 2)},
        {"s": (N[0] // 2 + 1, N[0] + 2)},
        {"axes": (0,)},
        {"axes": (1,)},
        {"axes": (-1,)},
        {"axes": (-2,)},
        {"axes": (0, 1)},
        {"axes": (1, 0)},
        {"axes": (1, 0, 1)},
    )

    for kwargs in all_kwargs:
        print(f"=== 2D R2C {dtype}, args: {kwargs} ===")
        out = np.fft.rfft2(Z, **kwargs)
        out_num = num.fft.rfft2(Z_num, **kwargs)
        assert allclose(out, out_num)
        out = np.fft.rfft2(np.swapaxes(Z, 0, 1), **kwargs)
        out_num = num.fft.rfft2(np.swapaxes(Z_num, 0, 1), **kwargs)
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
    Z = np.random.rand(*N).astype(dtype)
    Z_num = num.array(Z)

    all_kwargs = (
        (
            {},
            {"norm": "forward"},
            {"norm": "ortho"},
            {"s": (N[0] - 1, N[1] - 2, N[2] // 2)},
            {"s": (N[0] + 1, N[1] + 2, N[2] + 3)},
        )
        + tuple({"axes": (i,)} for i in range(3))
        + tuple({"axes": (-i,)} for i in range(1, 4))
        + tuple({"axes": (i + 1, i)} for i in range(2))
        + ({"axes": (0, 2, 1, 1, -1)},)
    )

    for kwargs in all_kwargs:
        print(f"=== 3D R2C {dtype}, args: {kwargs} ===")
        out = np.fft.rfftn(Z, **kwargs)
        out_num = num.fft.rfftn(Z_num, **kwargs)
        assert allclose(out, out_num)
        out = np.fft.rfftn(np.swapaxes(Z, 0, 1), **kwargs)
        out_num = num.fft.rfftn(np.swapaxes(Z_num, 0, 1), **kwargs)
        assert allclose(out, out_num)
        out = np.fft.rfftn(np.swapaxes(Z, 2, 1), **kwargs)
        out_num = num.fft.rfftn(np.swapaxes(Z_num, 2, 1), **kwargs)
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


def test_1d():
    check_1d_r2c(N=153)
    check_1d_r2c(N=153, dtype=np.float32)


def test_2d():
    check_2d_r2c(N=(28, 10))
    check_2d_r2c(N=(28, 10), dtype=np.float32)


def test_3d():
    check_3d_r2c(N=(6, 10, 12))
    check_3d_r2c(N=(6, 10, 12), dtype=np.float32)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
