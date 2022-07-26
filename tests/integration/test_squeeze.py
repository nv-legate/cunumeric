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

x = np.array([[[1, 2, 3]]])


def test_default():
    y = x.squeeze()

    assert np.array_equal(y, [1, 2, 3])


def test_axis_1d():
    y = x.squeeze(axis=1)

    assert np.array_equal(y, [[1, 2, 3]])


def test_axis_2d():
    x = np.array([[[1], [2], [3]]])

    y = x.squeeze(axis=(0, 2))

    assert np.array_equal(y, [1, 2, 3])


def test_idempotent():
    x = np.array([1, 2, 3])

    y = x.squeeze()

    assert x is y


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
