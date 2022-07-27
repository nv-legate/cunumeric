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
    x = num.array([1.0, 2.0, 3.0, 4.0])
    y = num.exp(x)

    # not using utils.comparison.allclose intentionally
    assert np.allclose(y, np.exp([1, 2, 3, 4]))
    assert num.allclose(y, np.exp([1, 2, 3, 4]))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
