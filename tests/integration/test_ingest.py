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
import pytest
from legate.core import (
    CustomSplit,
    Rect,
    TiledSplit,
    float64,
    get_legion_context,
    get_legion_runtime,
    ingest,
    legion,
)

import cunumeric as num

tile_shape = (4, 7)
colors = (5, 3)
shape = tuple(ci * di for (ci, di) in zip(colors, tile_shape))


def get_subdomain(color):
    return Rect(
        lo=[ci * di for (ci, di) in zip(color, tile_shape)],
        hi=[(ci + 1) * di for (ci, di) in zip(color, tile_shape)],
    )


def get_buffer(color):
    arr = np.zeros(tile_shape)
    base = float(
        color[0] * tile_shape[0] * shape[1] + color[1] * tile_shape[1]
    )
    for i in range(tile_shape[0]):
        for j in range(tile_shape[1]):
            arr[i, j] = base + shape[1] * i + j
    return arr.data


def get_local_colors():
    num_shards = legion.legion_runtime_total_shards(
        get_legion_runtime(), get_legion_context()
    )
    shard = legion.legion_runtime_local_shard(
        get_legion_runtime(), get_legion_context()
    )
    res = []
    i = 0
    for color in Rect(colors):
        if i % num_shards == shard:
            res.append(color)
        i += 1
    return res


def _ingest(custom_partitioning, custom_sharding):
    data_split = (
        CustomSplit(get_subdomain)
        if custom_partitioning
        else TiledSplit(tile_shape)
    )
    tab = ingest(
        float64,
        shape,
        colors,
        data_split,
        get_buffer,
        get_local_colors if custom_sharding else None,
    )
    return num.array(tab)


@pytest.mark.parametrize("custom_sharding", [True, False])
@pytest.mark.parametrize("custom_partitioning", [True, False])
def test(custom_partitioning, custom_sharding):
    size = 1
    for d in shape:
        size *= d
    a_np = np.arange(size).reshape(shape)
    a_num = _ingest(custom_partitioning, custom_sharding)
    assert np.array_equal(a_np, a_num)
    assert np.array_equal(a_np, a_num * 1.0)  # force a copy


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
