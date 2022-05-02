# Copyright 2022 NVIDIA Corporation
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

import cunumeric as num

fns = ["bartlett", "blackman", "hamming", "hanning"]


def test():
    for fn in fns:
        print(f"Testing cunumeric.{fn}")
        np_fn = getattr(np, fn)
        num_fn = getattr(num, fn)

        out_np = np_fn(0)
        out_num = num_fn(0)

        assert np.allclose(out_np, out_num)

        out_np = np_fn(1)
        out_num = num_fn(1)

        assert np.allclose(out_np, out_num)

        out_np = np_fn(10)
        out_num = num_fn(10)

        assert np.allclose(out_np, out_num)

        out_np = np_fn(100)
        out_num = num_fn(100)

        assert np.allclose(out_np, out_num)

    print("Testing cunumeric.kaiser")
    out_np = np.kaiser(0, 0)
    out_num = num.kaiser(0, 0)

    assert np.allclose(out_np, out_num)

    out_np = np.kaiser(1, 0)
    out_num = num.kaiser(1, 0)

    assert np.allclose(out_np, out_num)
    out_np = np.kaiser(10, 0)
    out_num = num.kaiser(10, 0)

    assert np.allclose(out_np, out_num)

    out_np = np.kaiser(100, 6)
    out_num = num.kaiser(100, 6)

    assert np.allclose(out_np, out_num)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
