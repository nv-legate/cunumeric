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

from benchmark import run_benchmark, time

import cunumeric as np

float_type = "float32"

# A simplified implementation of Richardson-Lucy deconvolution


def run_richardson_lucy(shape, filter_shape, num_iter, warmup, timing):
    image = np.random.rand(*shape).astype(float_type)
    psf = np.random.rand(*filter_shape).astype(float_type)
    im_deconv = np.full(image.shape, 0.5, dtype=float_type)
    psf_mirror = np.flip(psf)

    start = time()

    for idx in range(num_iter + warmup):
        if idx == warmup:
            start = time()
        conv = np.convolve(im_deconv, psf, mode="same")
        relative_blur = image / conv
        im_deconv *= np.convolve(relative_blur, psf_mirror, mode="same")

    stop = time()
    total = (stop - start) / 1000.0
    if timing:
        print("Elapsed Time: " + str(total) + " ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=10,
        dest="I",
        help="number of iterations to run",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=1,
        dest="warmup",
        help="warm-up iterations",
    )
    parser.add_argument(
        "-x",
        type=int,
        default=20,
        dest="X",
        help="number of elements in X dimension",
    )
    parser.add_argument(
        "-y",
        type=int,
        default=20,
        dest="Y",
        help="number of elements in Y dimension",
    )
    parser.add_argument(
        "-z",
        type=int,
        default=20,
        dest="Z",
        help="number of elements in Z dimension",
    )
    parser.add_argument(
        "-fx",
        type=int,
        default=4,
        dest="FX",
        help="number of filter weights in X dimension",
    )
    parser.add_argument(
        "-fy",
        type=int,
        default=4,
        dest="FY",
        help="number of filter weights in Y dimension",
    )
    parser.add_argument(
        "-fz",
        type=int,
        default=4,
        dest="FZ",
        help="number of filter weights in Z dimension",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 "
        "- normal execution)",
    )
    args = parser.parse_args()
    run_benchmark(
        run_richardson_lucy,
        args.benchmark,
        "Richardson Lucy",
        (
            (args.X, args.Y, args.Z),
            (args.FX, args.FY, args.FZ),
            args.I,
            args.warmup,
            args.timing,
        ),
    )
