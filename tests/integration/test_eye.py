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


def idk(k):
    return f"k={k}"


@pytest.mark.parametrize("k", KS, ids=idk)
def test_square(k):
    print(f"np.eye(5, k={k})")
    e_lg = num.eye(5, k=k)
    e_np = np.eye(5, k=k)
    assert np.array_equal(e_lg, e_np)


@pytest.mark.parametrize("k", KS, ids=idk)
def test_wide(k):
    print(f"np.eye(5, 6, k={k})")
    e_lg = num.eye(5, 6, k=k)
    e_np = np.eye(5, 6, k=k)
    assert np.array_equal(e_lg, e_np)


@pytest.mark.parametrize("k", KS, ids=idk)
def test_tall(k):
    print(f"np.eye(5, 4, k={k})")
    e_lg = num.eye(5, 4, k=k)
    e_np = np.eye(5, 4, k=k)
    assert np.array_equal(e_lg, e_np)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
