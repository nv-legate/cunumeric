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
from utils.random import assert_distribution

import cunumeric as num


def test_randn():
    # We cannot expect that random generation will match with NumPy's,
    # even if initialized with the same seed, so all we can do to verify
    # the results is check that they have the expected distribution.
    a_num = num.random.randn(10000)
    # `mean(a_num)` itself will be normally distributed, with:
    # mean = population mean = 0.0
    # stddev = population stddev / sqrt(samples) = 0.01
    # so a range of -0.05 to 0.05 represents 5 standard deviations
    # which should be extremely unlikely to run over
    assert_distribution(a_num, 0.0, 1.0, mean_tol=0.05)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
