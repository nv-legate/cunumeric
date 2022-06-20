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


def compare_assert(a_np, a_num):
    if not num.allclose(a_np, a_num):
        print("numpy, shape " + str(a_np.shape) + ":")
        print(a_np)
        print("cuNumeric, shape " + str(a_num.shape) + ":")
        print(a_num)
        assert False


def generate_random(shape, datatype):
    print("Generate random for " + str(datatype))
    a_np = None
    volume = 1
    for i in shape:
        volume *= i

    if np.issubdtype(datatype, np.integer):
        a_np = np.array(
            np.random.randint(
                np.iinfo(datatype).min, np.iinfo(datatype).max, size=volume
            ),
            dtype=datatype,
        )
    elif np.issubdtype(datatype, np.floating):
        a_np = np.array(np.random.random(size=volume), dtype=datatype)
    elif np.issubdtype(datatype, np.complexfloating):
        a_np = np.array(
            np.random.random(size=volume) + np.random.random(size=volume) * 1j,
            dtype=datatype,
        )
    else:
        print("UNKNOWN type " + str(datatype))
        assert False
    return a_np.reshape(shape)


def check_api(a=None):
    if a is None:
        a = np.arange(24)

    a_sorted = np.sort(a)
    a_argsorted = np.argsort(a)
    v = generate_random((10,), a.dtype)
    v_scalar = v[0]
    print("A=")
    print(a)
    print("A_sorted=")
    print(a_sorted)
    print("A_argsorted=")
    print(a_argsorted)
    print("V=")
    print(v)

    a_num = num.array(a)
    compare_assert(a, a_num)

    v_num = num.array(v)
    compare_assert(v, v_num)

    a_num_sorted = num.array(a_sorted)
    compare_assert(a_sorted, a_num_sorted)

    a_num_argsorted = num.array(a_argsorted)
    compare_assert(a_argsorted, a_num_argsorted)

    # left, scalar
    print("Left/Scalar")
    compare_assert(
        a_sorted.searchsorted(v_scalar), a_num_sorted.searchsorted(v_scalar)
    )
    # left, array
    print("Left/Array")
    compare_assert(a_sorted.searchsorted(v), a_num_sorted.searchsorted(v_num))
    # left, array, sorter
    print("Left/Array/Sorter")
    compare_assert(
        a.searchsorted(v, sorter=a_argsorted),
        a_num.searchsorted(v_num, sorter=a_num_argsorted),
    )

    # right, scalar
    print("Right/Scalar")
    compare_assert(
        a_sorted.searchsorted(v_scalar, side="right"),
        a_num_sorted.searchsorted(v_scalar, side="right"),
    )
    # right, array
    print("Right/Array")
    compare_assert(
        a_sorted.searchsorted(v, side="right"),
        a_num_sorted.searchsorted(v_num, side="right"),
    )
    # right, array, sorter
    print("Right/Array/Sorter")
    compare_assert(
        a.searchsorted(v, side="right", sorter=a_argsorted),
        a_num.searchsorted(v_num, side="right", sorter=a_num_argsorted),
    )


def check_dtypes():
    np.random.seed(42)
    check_api(generate_random((156,), np.uint8))
    check_api(generate_random((123,), np.uint16))
    check_api(generate_random((241,), np.uint32))
    check_api(generate_random((1,), np.uint32))

    check_api(generate_random((21,), np.int8))
    check_api(generate_random((5,), np.int16))
    check_api(generate_random((34,), np.int32))
    check_api(generate_random((11,), np.int64))

    check_api(generate_random((31,), np.float32))
    check_api(generate_random((11,), np.float64))
    check_api(generate_random((422,), np.double))
    check_api(generate_random((220,), np.double))

    check_api(generate_random((244,), np.complex64))
    check_api(generate_random((24,), np.complex128))
    check_api(generate_random((220,), np.complex128))


def test():
    print("\n\n -----------  API test --------------\n")
    check_api()
    print("\n\n -----------  dtype test ------------\n")
    check_dtypes()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
