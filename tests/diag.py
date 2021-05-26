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

import legate.numpy as lg


def test():
    for k in [0, -1, 1, -2, 2]:
        print(f"diag(k={k})")
        a = lg.array(
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

        b = lg.diag(a, k=k)
        bn = np.diag(an, k=k)
        assert np.array_equal(b, bn)

        c = lg.diag(b, k=k)
        cn = np.diag(bn, k=k)
        assert np.array_equal(c, cn)

        d = lg.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ]
        )
        dn = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ]
        )

        e = lg.diag(d, k=k)
        en = np.diag(dn, k=k)
        assert np.array_equal(e, en)

        f = lg.diag(e, k=k)
        fn = np.diag(en, k=k)
        assert np.array_equal(f, fn)

    return


if __name__ == "__main__":
    test()
