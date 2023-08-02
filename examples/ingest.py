#!/usr/bin/env python

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

import argparse
import os
from glob import glob

import tifffile as tfl
from legate.core import CustomSplit, Rect, TiledSplit, ingest, uint16

import cunumeric as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "scans_dir",
)
parser.add_argument(
    "-c",
    "--colors",
    type=int,
    nargs="+",
    default=[3, 2, 2, 1],
)
parser.add_argument("--custom-partitioning", action="store_true")
parser.add_argument("--custom-sharding", action="store_true")
args = parser.parse_args()
dtype = uint16
tile_shape = (1, 301, 704, 360)
colors = tuple(args.colors)
shape = tuple(ci * di for (ci, di) in zip(colors, tile_shape))


def get_subdomain(color):
    return Rect(
        lo=[ci * di for (ci, di) in zip(color, tile_shape)],
        hi=[(ci + 1) * di for (ci, di) in zip(color, tile_shape)],
    )


def get_buffer(color):
    (channel, xtile, ytile, ztile) = color
    fnames = glob(
        os.path.join(
            args.scans_dir,
            "Scan1_NC_Iter_0_"
            + f"ch{channel}_*_{xtile:03}x_{ytile:03}y_{ztile:03}z_0000t.tif",
        )
    )
    assert len(fnames) == 1
    im = tfl.imread(fnames[0])
    assert tile_shape == (1,) + im.shape  # channel dimension is implicit
    return im.reshape(tile_shape).data


def get_local_colors():
    # Assumes we were launched using mpirun
    num_ranks = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    res = []
    i = 0
    for color in Rect(colors):
        if i % num_ranks == rank:
            res.append(color)
        i += 1
    return res


data_split = (
    CustomSplit(get_subdomain)
    if args.custom_partitioning
    else TiledSplit(tile_shape)
)
tab = ingest(
    dtype,
    shape,
    colors,
    data_split,
    get_buffer,
    get_local_colors if args.custom_sharding else None,
)
arr = np.array(tab)
b = arr + arr
