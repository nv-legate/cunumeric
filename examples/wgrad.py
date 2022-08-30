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

from legate.timing import time

import cunumeric as np


def initialize(C, K, B, H, W):
    x = np.random.randn(C, B, H, W)
    y = np.random.randn(K, B, H, W)
    return x, y


def cross_correlate(x, y, C, K, R, S, B, H, W):
    dw = np.zeros(shape=(R, S, C, K))
    # cross-correlate images to compute weight gradients
    y_pad = np.zeros(shape=(K, B, H + R - 1, W + S - 1))
    y_pad[:, :, R / 2 : -(R / 2), S / 2 : -(S / 2)] = y
    for r in range(R):
        for s in range(S):
            y_shift = y_pad[:, :, r : r + H, s : s + W]
            for c in range(C):
                for k in range(K):
                    dw[r, s, c, k] = np.sum(
                        x[c, :, :, :] * y_shift[k, :, :, :]
                    )
    return dw


def run_wgrad(H=256, W=256, B=32, C=256, K=32, R=5, S=5, timing=False):
    start = time()
    x, y = initialize(C, K, B, H, W)
    _ = cross_correlate(x, y, C, K, R, S, B, H, W)
    stop = time()
    total = (stop - start) / 1000.0
    if timing:
        print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", type=int, default=32, dest="B", help="batch size"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        dest="H",
        help="height of images in pixels",
    )
    parser.add_argument(
        "-i", "--input", type=int, default=256, dest="C", help="input channels"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=int,
        default=32,
        dest="K",
        help="output channels",
    )
    parser.add_argument(
        "-r",
        "--radix",
        type=int,
        default=5,
        dest="R",
        help="convolution radix",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=256,
        dest="W",
        help="width of images in pixels",
    )
    args = parser.parse_args(parser)
    run_wgrad(
        args.H, args.W, args.B, args.C, args.K, args.R, args.R, args.timing
    )
