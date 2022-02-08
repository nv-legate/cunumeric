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

import cunumeric as num


def test():
    for f in ["tril", "triu"]:
        num_f = getattr(num, f)
        np_f = getattr(np, f)
        for k in [0, -1, 1, -2, 2]:
            print(f"{f}(k={k})")
            a = num.array(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                    [17, 18, 19, 20],
                ]
            )
            an = np.array(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                    [17, 18, 19, 20],
                ]
            )

            b = num_f(a, k=k)
            bn = np_f(an, k=k)
            assert num.array_equal(b, bn)

            b = num_f(a[0, :], k=k)
            bn = np_f(an[0, :], k=k)
            assert num.array_equal(b, bn)


if __name__ == "__main__":
    test()
