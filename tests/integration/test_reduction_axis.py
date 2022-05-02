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

from itertools import permutations

import numpy as np
import pytest

import cunumeric as cn


def _sum(shape, axis, lib):
    return lib.ones(shape).sum(axis=axis)


# Try various non-square shapes, to nudge the core towards trying many
# different partitionings.
@pytest.mark.parametrize("axis", range(3), ids=str)
@pytest.mark.parametrize("shape", permutations((3, 4, 5)), ids=str)
def test_3d(shape, axis):
    assert np.array_equal(_sum(shape, axis, np), _sum(shape, axis, cn))


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
