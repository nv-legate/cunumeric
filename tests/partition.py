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

import math

import numpy as np

import cunumeric as num


def assert_partition(a_num, kth, axis):

    # compute volume
    shape = a_num.shape
    volume = 1
    for i in range(a_num.ndim):
        volume *= shape[i]

    # move axis to end and flatten other
    sort_dim = shape[axis]
    flattened = (int)(volume / sort_dim)
    a_mod = a_num.swapaxes(axis, a_num.ndim - 1)
    a_mod = a_mod.reshape((flattened, sort_dim))

    for segment in range(flattened):
        if kth > 0:
            left_part = a_mod[segment, 0:kth]
            max_left = left_part.max()
            if max_left > a_mod[segment, kth]:
                print(a_num)
                assert False
        if kth < sort_dim:
            right_part = a_mod[segment, kth:sort_dim]
            min_right = right_part.min()
            if min_right < a_mod[segment, kth]:
                print(a_num)
                assert False


def assert_argpartition(a_num, a_org, kth, axis):
    assert a_num.dtype == np.int64
    assert a_num.ndim == a_org.ndim

    # compute volume
    shape = a_num.shape
    shape_org = a_org.shape
    for i in range(a_num.ndim):
        assert shape[i] == shape_org[i]

    a_reindexed = np.take_along_axis(a_org, a_num, axis=axis)

    assert_partition(a_reindexed, kth, axis)


def test_api(a=None):
    if a is None:
        a = np.arange(4 * 2 * 3).reshape(4, 2, 3)
    a_num = num.array(a)

    shape = a.shape
    volume = 1
    for i in range(a.ndim):
        volume *= shape[i]

    # partition axes
    for i in range(a.ndim):
        kth = math.floor(shape[i] / 2)
        print("partition axis " + str(i))
        assert_partition(
            num.partition(a_num, kth=kth, axis=i).__array__(), kth, i
        )

    # flatten
    print("partition flattened")
    kth = math.floor(volume / 2)
    assert_partition(
        num.partition(a_num, kth=kth, axis=None).__array__(), kth, 0
    )

    # in-place partition
    kth = math.floor(shape[a.ndim - 1] / 2)
    copy_a_num = a_num.copy()
    copy_a_num.partition(kth)
    assert_partition(copy_a_num.__array__(), kth, a.ndim - 1)

    # argpartition
    for i in range(a.ndim):
        kth = math.floor(shape[i] / 2)
        print("argpartition axis " + str(i))
        assert_argpartition(
            num.argpartition(a_num, kth, axis=i).__array__(),
            a_num.__array__(),
            kth,
            i,
        )

    # flatten
    print("argpartition flattened")
    kth = math.floor(volume / 2)
    assert_argpartition(
        num.argpartition(a_num, kth, axis=None).__array__(),
        a_num.flatten().__array__(),
        kth,
        0,
    )

    # nd.argpartition -- no change to array
    kth = math.floor(shape[a.ndim - 1] / 2)
    copy_a_num = a_num.copy()
    assert_argpartition(
        copy_a_num.argpartition(kth).__array__(),
        a_num.__array__(),
        kth,
        a.ndim - 1,
    )
    assert num.allclose(copy_a_num, copy_a_num)


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


def test_dtypes():
    np.random.seed(42)
    test_api(generate_random((2, 5, 7), np.uint8))
    test_api(generate_random((8, 5), np.uint16))
    test_api(generate_random((22, 5, 7), np.uint32))
    test_api(generate_random((220,), np.uint32))

    test_api(generate_random((2, 5, 7), np.int8))
    test_api(generate_random((8, 5), np.int16))
    test_api(generate_random((22, 5, 7), np.int32))
    test_api(generate_random((2, 5, 7), np.int64))

    test_api(generate_random((8, 5), np.float32))
    test_api(generate_random((8, 5), np.float64))
    test_api(generate_random((22, 5, 7), np.double))
    test_api(generate_random((220,), np.double))

    test_api(generate_random((2, 5, 7), np.complex64))
    test_api(generate_random((2, 5, 7), np.complex128))
    test_api(generate_random((220,), np.complex128))


def test():
    print("\n\n -----------  dtype test ------------\n")
    test_dtypes()


if __name__ == "__main__":
    test()
