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


def gen_result(lib):
    # Try various non-square shapes, to nudge the core towards trying many
    # different partitionings.
    for shape in permutations((3, 4, 5)):
        x = lib.ones(shape)
        for axis in range(len(shape)):
            yield x.sum(axis=axis)


def test():
    for (np_res, cn_res) in zip(gen_result(np), gen_result(cn)):
        assert np.array_equal(np_res, cn_res)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
