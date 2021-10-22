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

import cunumeric as num


def test(n):
    for dtype in [np.int64, np.int32, np.int16]:
        print(dtype)
        v_num = num.random.randint(0, 9, size=n, dtype=dtype)
        w_num = num.random.randn(n)

        v_np = v_num.__array__()
        w_np = w_num.__array__()

        out_np = np.bincount(v_np)
        out_num = num.bincount(v_num)
        assert num.array_equal(out_np, out_num)

        out_np = np.bincount(v_np, weights=w_np)
        out_num = num.bincount(v_num, weights=w_num)
        assert num.allclose(out_np, out_num)

    return


if __name__ == "__main__":
    test(8000)
