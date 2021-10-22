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

import cunumeric as lg


def test(n):
    for dtype in [np.int64, np.int32, np.int16]:
        print(dtype)
        v_lg = lg.random.randint(0, 9, size=n, dtype=dtype)
        w_lg = lg.random.randn(n)

        v_np = v_lg.__array__()
        w_np = w_lg.__array__()

        out_np = np.bincount(v_np)
        out_lg = lg.bincount(v_lg)
        assert lg.array_equal(out_np, out_lg)

        out_np = np.bincount(v_np, weights=w_np)
        out_lg = lg.bincount(v_lg, weights=w_lg)
        assert lg.allclose(out_np, out_lg)

    return


if __name__ == "__main__":
    test(8000)
