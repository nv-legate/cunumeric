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
from cunumeric.utils import dot_modes
from test_tools.generators import mk_0to1_array

import cunumeric as cn
from legate.core import LEGATE_MAX_DIM


def gen_shapes(a_modes, b_modes):
    yield ((5,) * len(a_modes), (5,) * len(b_modes))
    for mode_to_squeeze in set(a_modes + b_modes):
        a_shape = tuple((1 if m == mode_to_squeeze else 5) for m in a_modes)
        b_shape = tuple((1 if m == mode_to_squeeze else 5) for m in b_modes)
        yield (a_shape, b_shape)


def gen_inputs_of_various_shapes(lib, modes):
    # 5x5x...x5, with up to one dimension set to 1
    # making sure common modes appear with the same extent on both arrays
    (a_modes, b_modes, out_modes) = modes
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


def gen_transposed_inputs(lib, modes):
    (a_modes, b_modes, out_modes) = modes
    a = mk_0to1_array(lib, (5,) * len(a_modes))
    b = mk_0to1_array(lib, (5,) * len(b_modes))
    for a_axes in gen_permutations(len(a_modes)):
        for b_axes in gen_permutations(len(b_modes)):
            if lib == cn:
                print(f"  transpose{a_axes} x transpose{b_axes}")
            yield (a.transpose(a_axes), b.transpose(b_axes))


def gen_typed_inputs(lib, modes):
    (a_modes, b_modes, out_modes) = modes
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


def gen_typed_output(lib, modes, a, b):
    (a_modes, b_modes, out_modes) = modes
    out_shape = (5,) * len(out_modes)
    out_dtype = np.float32
    if lib == cn:
        print(f"  -> {out_dtype}")
    yield lib.zeros(out_shape, out_dtype)


def _test_contraction(modes, operation, gen_inputs, gen_output=None):
    for (np_inputs, cn_inputs) in zip(
        gen_inputs(np, modes), gen_inputs(cn, modes)
    ):
        np_res = operation(np, *np_inputs)
        cn_res = operation(cn, *cn_inputs)
        rtol = (
            2e-03 if any(x.dtype == np.float16 for x in np_inputs) else 1e-05
        )
        assert np.allclose(np_res, cn_res, rtol=rtol)
        if gen_output is not None:
            for cn_out in gen_output(cn, modes, *cn_inputs):
                operation(cn, *cn_inputs, out=cn_out)
                rtol = (
                    2e-03
                    if any(x.dtype == np.float16 for x in np_inputs)
                    or cn_out.dtype == np.float16
                    else 1e-05
                )
                assert np.allclose(cn_out, cn_res, rtol=rtol)


def test_contraction(name, modes, operation):
    (a_modes, b_modes, out_modes) = modes
    if len(set(a_modes) | set(b_modes)) > LEGATE_MAX_DIM:
        # Total number of distinct modes can't exceed maximum Legion dimension,
        # because we may need to promote arrays so that one includes all modes.
        return
    print(f"testing {name} (various shapes)")
    _test_contraction(modes, operation, gen_inputs_of_various_shapes)
    print(f"testing {name} (permutations)")
    _test_contraction(modes, operation, gen_transposed_inputs)
    print(f"testing {name} (casting & out=)")
    _test_contraction(modes, operation, gen_typed_inputs, gen_typed_output)


def run_dot(lib, *args, **kwargs):
    return lib.dot(*args, **kwargs)


def test():
    for a_ndim in range(LEGATE_MAX_DIM + 1):
        for b_ndim in range(LEGATE_MAX_DIM + 1):
            name = f"dot({a_ndim} x {b_ndim})"
            modes = dot_modes(a_ndim, b_ndim)
            test_contraction(name, modes, run_dot)


if __name__ == "__main__":
    test()
