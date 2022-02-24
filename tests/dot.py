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

from string import ascii_lowercase, ascii_uppercase

import numpy as np
from test_tools.generators import mk_0to1_array

import cunumeric as cn
from legate.core import LEGATE_MAX_DIM


def gen_shapes(a_modes, b_modes):
    yield ((5,) * len(a_modes), (5,) * len(b_modes))
    for mode_to_squeeze in set(a_modes + b_modes):
        a_shape = tuple((1 if m == mode_to_squeeze else 5) for m in a_modes)
        b_shape = tuple((1 if m == mode_to_squeeze else 5) for m in b_modes)
        yield (a_shape, b_shape)


def gen_inputs_of_various_shapes(lib, a_modes, b_modes):
    for (a_shape, b_shape) in gen_shapes(a_modes, b_modes):
        if lib == cn:
            print(f"  {a_shape} x {b_shape}")
        yield (mk_0to1_array(lib, a_shape), mk_0to1_array(lib, b_shape))


def gen_permutations(ndim):
    yield tuple(range(ndim))  # e.g. (0, 1, 2, 3)
    if ndim > 1:
        yield tuple(range(ndim - 1, -1, -1))  # e.g. (3, 2, 1, 0)
    if ndim > 2:
        yield (
            tuple(range(ndim // 2)) + tuple(range(ndim - 1, ndim // 2 - 1, -1))
        )  # e.g. (0, 1, 3, 2)


def gen_transposed_inputs(lib, a_modes, b_modes):
    a = mk_0to1_array(lib, (5,) * len(a_modes))
    b = mk_0to1_array(lib, (5,) * len(b_modes))
    for a_axes in gen_permutations(len(a_modes)):
        for b_axes in gen_permutations(len(b_modes)):
            if lib == cn:
                print(f"  transpose{a_axes} x transpose{b_axes}")
            yield (a.transpose(a_axes), b.transpose(b_axes))


def gen_typed_inputs(lib, a_modes, b_modes):
    a_shape = (5,) * len(a_modes)
    b_shape = (5,) * len(b_modes)
    for (a_dtype, b_dtype) in [
        (np.float16, np.float16),
        (np.float16, np.float32),
        (np.float32, np.float32),
        (np.complex64, np.complex64),
    ]:
        if lib == cn:
            print(f"  {a_dtype} x {b_dtype}")
        yield (
            mk_0to1_array(lib, a_shape, a_dtype),
            mk_0to1_array(lib, b_shape, b_dtype),
        )


def gen_typed_output(lib, a_modes, b_modes):
    if len(a_modes) == 0:
        out_ndim = len(b_modes)
    elif len(b_modes) == 0:
        out_ndim = len(a_modes)
    else:
        out_ndim = len(a_modes) + len(b_modes) - 2
    for out_dtype in [np.float32]:
        if lib == cn:
            print(f"  -> {out_dtype}")
        yield lib.zeros((5,) * out_ndim, out_dtype)


def test_np_vs_cn(a_modes, b_modes, gen_inputs, gen_output=None):
    for (np_inputs, cn_inputs) in zip(
        gen_inputs(np, a_modes, b_modes),
        gen_inputs(cn, a_modes, b_modes),
    ):
        np_res = np.dot(*np_inputs)
        cn_res = cn.dot(*cn_inputs)
        rtol = (
            2e-03 if any(x.dtype == np.float16 for x in np_inputs) else 1e-05
        )
        assert np.allclose(np_res, cn_res, rtol=rtol)
        if gen_output is not None:
            for cn_out in gen_output(cn, a_modes, b_modes):
                cn.dot(*cn_inputs, out=cn_out)
                rtol = (
                    2e-03
                    if any(x.dtype == np.float16 for x in np_inputs)
                    or cn_out.dtype == np.float16
                    else 1e-05
                )
                assert np.allclose(cn_out, cn_res, rtol=rtol)


def test():
    for a_ndim in range(LEGATE_MAX_DIM + 1):
        for b_ndim in range(LEGATE_MAX_DIM + 1):
            if a_ndim + b_ndim - 1 > LEGATE_MAX_DIM:
                # Total number of distinct modes can't exceed maximum Legion
                # dimension, because we may need to promote arrays so that
                # each one includes all modes.
                continue
            a_modes = list(ascii_lowercase[:a_ndim])
            b_modes = list(ascii_uppercase[:b_ndim])
            if a_ndim >= 1 and b_ndim >= 1:
                b_modes[-1 if b_ndim == 1 else -2] = a_modes[-1]
            print(f"testing {a_ndim} x {b_ndim} (various shapes)")
            # (5x5x...x5, with up to one dimension set to 1)
            test_np_vs_cn(a_modes, b_modes, gen_inputs_of_various_shapes)
            print(f"testing {a_ndim} x {b_ndim} (permutations)")
            test_np_vs_cn(a_modes, b_modes, gen_transposed_inputs)
            if a_ndim <= 1 and b_ndim <= 2 or a_ndim <= 2 and b_ndim <= 1:
                print(f"testing {a_ndim} x {b_ndim} (casting & out=)")
                test_np_vs_cn(
                    a_modes, b_modes, gen_typed_inputs, gen_typed_output
                )
            elif a_ndim <= 2 and b_ndim <= 2:
                print(f"testing {a_ndim} x {b_ndim} (casting)")
                test_np_vs_cn(a_modes, b_modes, gen_typed_inputs)


if __name__ == "__main__":
    test()
