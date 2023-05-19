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

DIM_CASE = (5, 5)

FLAGS = [
    "c_contiguous",
    "f_contiguous",
    "writeable",
    "aligned",
    "writebackifcopy",
    "behaved",
    "carray",
    "farray",
]

ODSKIP = [pytest.param("owndata", marks=pytest.mark.skip)]
OSKIP = [pytest.param("O", marks=pytest.mark.skip)]

SHORT_FLAGS = ["C", "F", "W", "A", "X", "B", "CA", "FA"]


class Test_flags:
    @pytest.mark.parametrize("flag", FLAGS + ODSKIP)
    def test_default_attr(self, flag):
        npflags = np.zeros(shape=DIM_CASE).flags
        numflags = num.zeros(shape=DIM_CASE).flags

        assert getattr(npflags, flag) == getattr(numflags, flag)

    @pytest.mark.parametrize("flag", FLAGS + ODSKIP)
    def test_default_attr_view(self, flag):
        npflags = np.zeros(shape=DIM_CASE).view().flags
        numflags = num.zeros(shape=DIM_CASE).view().flags

        assert getattr(npflags, flag) == getattr(numflags, flag)

    @pytest.mark.parametrize("flag", SHORT_FLAGS + OSKIP)
    def test_default_item(self, flag):
        npflags = np.zeros(shape=DIM_CASE).flags
        numflags = num.zeros(shape=DIM_CASE).flags

        assert npflags[flag] == numflags[flag]

    @pytest.mark.parametrize("flag", SHORT_FLAGS + OSKIP)
    def test_default_item_view(self, flag):
        npflags = np.zeros(shape=DIM_CASE).view().flags
        numflags = num.zeros(shape=DIM_CASE).view().flags

        assert npflags[flag] == numflags[flag]


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
