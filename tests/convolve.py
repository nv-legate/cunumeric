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
import scipy.signal as sig

import cunumeric as num

shapes = ((100,), (10, 10), (10, 10, 10))
filter_shapes = ((5,), (3, 5), (3, 5, 3))


def test_convolve(a, v):
    anp = a.__array__()
    vnp = v.__array__()

    out = num.convolve(a, v, mode="same")
    if a.ndim > 1:
        out_np = sig.convolve(anp, vnp, mode="same")
    else:
        out_np = np.convolve(anp, vnp, mode="same")

    assert num.allclose(out, out_np)


def test_double():
    for shape, filter_shape in zip(shapes, filter_shapes):
        a = num.random.rand(*shape)
        v = num.random.rand(*filter_shape)

        test_convolve(a, v)
        test_convolve(v, a)


def test_int():
    for shape, filter_shape in zip(shapes, filter_shapes):
        a = num.random.randint(0, 5, shape)
        v = num.random.randint(0, 5, filter_shape)

        test_convolve(a, v)


if __name__ == "__main__":
    test_double()
    test_int()
