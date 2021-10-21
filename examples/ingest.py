#!/usr/bin/env python

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

import argparse
import os
from glob import glob

import pyarrow as pa
import tifffile as tfl

import legate.numpy as np
from legate.core import Rect, ingest

parser = argparse.ArgumentParser()
parser.add_argument(
    "scans_dir",
)
parser.add_argument(
    "--tile-shape",
    type=int,
    nargs="+",
    default=[1, 301, 704, 360],
)
parser.add_argument(
    "--colors",
    type=int,
    nargs="+",
    default=[3, 2, 2, 1],
)
args = parser.parse_args()
shape = [ci * di for (ci, di) in zip(args.colors, args.tile_shape)]


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
    assert args.tile_shape == [1] + list(im.shape)  # channel dim is implicit
    subdomain = Rect(
        lo=[ci * di for (ci, di) in zip(color, args.tile_shape)],
        hi=[(ci + 1) * di for (ci, di) in zip(color, args.tile_shape)],
    )
    return (subdomain, im.data)


def get_local_colors():
    # Assumes we were launched using mpirun
    num_ranks = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    res = []
    i = 0
    for color in args.colors:
        if i % num_ranks == rank:
            res.append(color)
        i += 1
    return res


# Legate-managed sharding
tab = ingest(pa.uint16(), shape, args.colors, get_buffer)
# Custom sharding
# tab = ingest(pa.uint16(), shape, args.colors, get_buffer, get_local_colors)

arr = np.array(tab)
b = arr + arr
