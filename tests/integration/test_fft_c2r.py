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


def check_1d_c2r(N, dtype=np.float64):
    Z = np.random.rand(N).astype(dtype) + np.random.rand(N).astype(dtype) * 1j
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
        print(f"=== 1D C2R {dtype}, args: {kwargs} ===")
        out = np.fft.irfft(Z, **kwargs)
        out_num = num.fft.irfft(Z_num, **kwargs)
        assert allclose(out, out_num)

    # Odd types
    out = np.fft.rfft(Z)
    out_num = num.fft.rfft(Z_num)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z)
    out_num = num.fft.ihfft(Z_num)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def check_2d_c2r(N, dtype=np.float64):
    Z = (
        np.random.rand(*N).astype(dtype)
        + np.random.rand(*N).astype(dtype) * 1j
    )
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
        print(f"=== 2D C2R {dtype}, args: {kwargs} ===")
        out = np.fft.irfft2(Z, **kwargs)
        out_num = num.fft.irfft2(Z_num, **kwargs)
        assert allclose(out, out_num)
        out = np.fft.irfft2(np.swapaxes(Z, 0, 1), **kwargs)
        out_num = num.fft.irfft2(np.swapaxes(Z_num, 0, 1), **kwargs)
        assert allclose(out, out_num)

    # Odd types
    out = np.fft.rfft2(Z)
    out_num = num.fft.rfft2(Z_num)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z)
    out_num = num.fft.ihfft(Z_num)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def check_3d_c2r(N, dtype=np.float64):
    print(f"\n=== 3D C2R {dtype}               ===")
    Z = (
        np.random.rand(*N).astype(dtype)
        + np.random.rand(*N).astype(dtype) * 1j
    )
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
        print(f"=== 3D C2R {dtype}, args: {kwargs} ===")
        out = np.fft.irfftn(Z, **kwargs)
        out_num = num.fft.irfftn(Z_num, **kwargs)
        assert allclose(out, out_num)
        out = np.fft.irfftn(np.swapaxes(Z, 0, 1), **kwargs)
        out_num = num.fft.irfftn(np.swapaxes(Z_num, 0, 1), **kwargs)
        assert allclose(out, out_num)
        out = np.fft.irfftn(np.swapaxes(Z, 2, 1), **kwargs)
        out_num = num.fft.irfftn(np.swapaxes(Z_num, 2, 1), **kwargs)
        assert allclose(out, out_num)

    # Odd types
    out = np.fft.rfftn(Z)
    out_num = num.fft.rfftn(Z_num)
    assert allclose(out, out_num)
    out = np.fft.ihfft(Z)
    out_num = num.fft.ihfft(Z_num)
    assert allclose(out, out_num)
    assert allclose(Z, Z_num)


def test_1d():
    check_1d_c2r(N=78)
    check_1d_c2r(N=78, dtype=np.float32)


def test_2d():
    check_2d_c2r(N=(28, 10))
    check_2d_c2r(N=(28, 10), dtype=np.float32)


def test_3d():
    check_3d_c2r(N=(6, 12, 10))
    check_3d_c2r(N=(6, 12, 10), dtype=np.float32)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
