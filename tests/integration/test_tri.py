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
import pytest

import cunumeric as num

KS = [0, -1, 1, -2, 2]


def _test(func, k):
    num_f = getattr(num, func)
    np_f = getattr(np, func)

    a = num_f(100, k=k)
    an = np_f(100, k=k)
    assert num.array_equal(a, an)

    a = num_f(100, 50, k=k)
    an = np_f(100, 50, k=k)
    assert num.array_equal(a, an)

    a = num_f(100, k=k, dtype=int)
    an = np_f(100, k=k, dtype=int)
    assert num.array_equal(a, an)

    a = num_f(100, k=k, dtype=bool)
    an = np_f(100, k=k, dtype=bool)
    assert num.array_equal(a, an)


@pytest.mark.parametrize("k", KS, ids=lambda k: f"(k={k})")
def test_tri(k):
    _test("tri", k)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
