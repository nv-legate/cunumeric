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

import cunumeric as num


def test():
    np.random.seed(13)
    anp = 20.0 * np.random.randn(4, 5) + 10.0
    a = num.array(anp)
    a_min = -10.0 * np.random.rand()
    a_max = 10.0 * np.random.rand()

    assert np.array_equal(
        num.clip(a, a_min, a_max, out=a), np.clip(anp, a_min, a_max, out=anp)
    )

    return


if __name__ == "__main__":
    test()
