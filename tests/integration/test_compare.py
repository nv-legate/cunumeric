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
import pytest

import cunumeric as np

x = pytest.fixture(lambda: np.array([1, 2, 3]))


def test_lt(x):
    y = x < 2
    assert np.array_equal(y, [True, False, False])


def test_le(x):
    y = x <= 2
    assert np.array_equal(y, [True, True, False])


def test_gt(x):
    y = x > 2
    assert np.array_equal(y, [False, False, True])


def test_ge(x):
    y = x >= 2
    assert np.array_equal(y, [False, True, True])


def test_eq(x):
    y = x == 2
    assert np.array_equal(y, [False, True, False])


# TODO (bev) why is this test in compare
def test_elementwise(x):
    y = (x + 2) * [6, 7, 8]
    assert np.array_equal(y, [18, 28, 40])


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
