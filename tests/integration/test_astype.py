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

TEST_VECTOR = [0, 0, 1, 2, 3, 0, 1, 2, 3]
ALL_BUT_COMPLEX = ["?", "b", "h", "i", "l", "B", "H", "I", "L", "e", "f", "d"]
ALL_TYPES = ALL_BUT_COMPLEX + ["F", "D"]
ORDER = ('C', 'F', 'A', 'K')
CASTING = ('no', 'equiv', 'safe', 'same_kind')


def to_dtype(s):
    return str(np.dtype(s))


def test_none():
    arr = None
    in_np = num.array(arr)
    msg = r"NoneType"
    with pytest.raises(TypeError, match=msg):
        in_np.astype("b")


@pytest.mark.parametrize("src_dtype", ALL_TYPES, ids=to_dtype)
def test_empty(src_dtype):
    arr = []
    in_np = np.array(arr)
    out_np = in_np.astype(src_dtype)

    in_num = np.array(arr)
    out_num = in_num.astype(src_dtype)

    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("src_dtype", ALL_BUT_COMPLEX, ids=to_dtype)
@pytest.mark.parametrize("dst_dtype", ALL_TYPES, ids=to_dtype)
def test_basic(src_dtype, dst_dtype):
    in_np = np.array(TEST_VECTOR, dtype=src_dtype)
    in_num = num.array(in_np)

    out_np = in_np.astype(dst_dtype)
    out_num = in_num.astype(dst_dtype)

    assert np.array_equal(out_num, out_np)


@pytest.mark.parametrize("src_dtype", ALL_BUT_COMPLEX, ids=to_dtype)
@pytest.mark.parametrize("dst_dtype", ALL_TYPES, ids=to_dtype)
@pytest.mark.parametrize("order", ORDER, ids=str)
def test_order(src_dtype, dst_dtype, order):
    in_np = np.array(TEST_VECTOR, dtype=src_dtype)
    in_num = num.array(in_np)

    out_np = in_np.astype(dst_dtype, order=order)
    out_num = in_num.astype(dst_dtype, order=order)

    assert np.array_equal(out_num, out_np)


@pytest.mark.parametrize("src_dtype", ALL_BUT_COMPLEX[1:5], ids=to_dtype)
@pytest.mark.parametrize("dst_dtype", ALL_BUT_COMPLEX[5:8], ids=to_dtype)
@pytest.mark.parametrize("cast", CASTING, ids=str)
def test_casting_negative(src_dtype, dst_dtype, cast):
    in_num = num.array(TEST_VECTOR, dtype=src_dtype)
    with pytest.raises(TypeError):
        in_num.astype(dst_dtype, casting=cast)


@pytest.mark.parametrize("src_dtype", ("F", "D"), ids=to_dtype)
@pytest.mark.parametrize("dst_dtype", ALL_TYPES[1:], ids=to_dtype)
def test_complex(src_dtype, dst_dtype):
    complex_input = [
        complex(v1, v2) for v1, v2 in zip(TEST_VECTOR[:-1], TEST_VECTOR[1:])
    ]
    in_np = np.array(complex_input, dtype=src_dtype)
    in_num = num.array(in_np)

    out_np = in_np.astype(dst_dtype)
    out_num = in_num.astype(dst_dtype)

    assert np.array_equal(out_num, out_np)


@pytest.mark.xfail
@pytest.mark.parametrize("src_dtype", ("F", "D"), ids=to_dtype)
def test_complex_negative(src_dtype):
    complex_input = [
        complex(v1, v2) for v1, v2 in zip(TEST_VECTOR[:-1], TEST_VECTOR[1:])
    ]
    in_np = np.array(complex_input, dtype=src_dtype)
    in_num = num.array(in_np)

    out_np = in_np.astype(to_dtype("?"))
    out_num = in_num.astype(to_dtype("?"))
    """
    Numpy and cuNumeric have different performance.
    For complex data 0.+1.j, Numpy set it as True, While cuNumeric set is as False.
    """
    assert np.array_equal(out_num, out_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
