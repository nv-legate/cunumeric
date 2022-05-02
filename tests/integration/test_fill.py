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


def test_basic():
    anp = np.random.randn(4, 5)
    bnp = np.full((4, 5), 13)
    a = num.array(anp)
    b = num.array(bnp)

    anp.fill(13)
    a.fill(13)

    assert np.array_equal(a, anp)

    bnp.fill(13)
    b.fill(13)

    assert np.array_equal(b, bnp)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
