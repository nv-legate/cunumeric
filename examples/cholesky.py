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


def cholesky(n, dtype):
    input = np.eye(n, dtype=dtype)

    start = time()
    np.cholesky(input, no_tril=True)
    stop = time()
    flops = (n**3) / 3 + 2 * n / 3
    print(f"{(stop - start) * 1e-3} ms, {flops / (stop - start) * 1e-3} GOP/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        dest="n",
        help="number of rows in the matrix",
    )
    parser.add_argument(
        "-t",
        "--type",
        default="float64",
        choices=["float32", "float64", "complex64", "complex128"],
        dest="dtype",
        help="data type",
    )
    args, unknown = parser.parse_known_args()
    cholesky(args.n, args.dtype)
