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
from test_tools.generators import scalar_gen

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


def nonscalar_gen(lib):
    for ndim in range(1, LEGATE_MAX_DIM):  # off-by-one is by design
        np.random.seed(42)
        yield lib.array(np.random.random_sample((5,) * ndim))


def tuple_set(tup, idx, val):
    lis = list(tup)
    lis[idx] = val
    return tuple(lis)


def array_gen(lib):
    # get single item from non-scalar array
    for arr in nonscalar_gen(lib):
        idx_tuple = arr.ndim * (2,)
        flat_idx = 0
        for (i, x) in enumerate(idx_tuple):
            flat_idx *= arr.shape[i]
            flat_idx += x
        yield arr[idx_tuple]
        yield arr.item(flat_idx)
        yield arr.item(idx_tuple)
        yield arr.item(*idx_tuple)
    # get single item from scalar array
    for arr in scalar_gen(lib, 42):
        idx_tuple = arr.ndim * (0,)
        yield arr[idx_tuple]
        yield arr.item()
        yield arr.item(0)
        yield arr.item(idx_tuple)
        yield arr.item(*idx_tuple)
    # get "multiple" items from scalar array
    for arr in scalar_gen(lib, 42):
        yield arr[arr.ndim * (slice(None),)]  # arr[:,:]
        # TODO: fix cunumeric#34
        # yield arr[arr.ndim * (slice(1, None),)] # arr[1:,1:]
    # set single item on non-scalar array
    for arr in nonscalar_gen(lib):
        idx_tuple = arr.ndim * (2,)
        arr[idx_tuple] = -1
        yield arr
    for arr in nonscalar_gen(lib):
        idx_tuple = arr.ndim * (2,)
        flat_idx = 0
        for (i, x) in enumerate(idx_tuple):
            flat_idx *= arr.shape[i]
            flat_idx += x
        arr.itemset(flat_idx, -1)
        yield arr
    for arr in nonscalar_gen(lib):
        idx_tuple = arr.ndim * (2,)
        arr.itemset(idx_tuple, -1)
        yield arr
    for arr in nonscalar_gen(lib):
        idx_tuple = arr.ndim * (2,)
        arr.itemset(*idx_tuple, -1)
        yield arr
    # set single item on scalar array
    for arr in scalar_gen(lib, 42):
        idx_tuple = arr.ndim * (0,)
        arr[idx_tuple] = -1
        yield arr
    for arr in scalar_gen(lib, 42):
        arr.itemset(-1)
        yield arr
    for arr in scalar_gen(lib, 42):
        arr.itemset(0, -1)
        yield arr
    for arr in scalar_gen(lib, 42):
        idx_tuple = arr.ndim * (0,)
        arr.itemset(idx_tuple, -1)
        yield arr
    for arr in scalar_gen(lib, 42):
        idx_tuple = arr.ndim * (0,)
        arr.itemset(*idx_tuple, -1)
        yield arr
    # set "multiple" items on scalar array
    for arr in scalar_gen(lib, 42):
        arr[arr.ndim * (slice(None),)] = -1  # arr[:,:] = -1
        yield arr
    # TODO: fix cunumeric#34
    # for arr in scalar_gen(lib, 42):
    #     arr[arr.ndim * (slice(1, None),)] = -1 # arr[1:,1:] = -1
    #     yield arr


def test():
    for (la, na) in zip(array_gen(num), array_gen(np)):
        assert np.array_equal(la, na)


if __name__ == "__main__":
    test()
