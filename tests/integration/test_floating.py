# Copyright 2022 NVIDIA Corporation
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

SHAPES = [
    (10,),
    (10, 20),
    (10, 20, 30),
]


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_modf(shape):
    x_np = np.random.random(shape)
    x_num = num.array(x_np)

    outs_np = np.modf(x_np)
    outs_num = num.modf(x_num)

    for out_np, out_num in zip(outs_np, outs_num):
        assert np.allclose(out_np, out_num)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_floating(shape):
    x_np = np.random.random(shape)
    x_num = num.array(x_np)

    frexp_np = np.frexp(x_np)
    frexp_num = num.frexp(x_num)

    for out_np, out_num in zip(frexp_np, frexp_num):
        assert np.allclose(out_np, out_num)

    ldexp_np = np.ldexp(*frexp_np)
    ldexp_num = np.ldexp(*frexp_num)

    assert np.allclose(ldexp_np, ldexp_num)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
