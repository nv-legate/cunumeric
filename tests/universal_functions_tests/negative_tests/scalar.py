# Copyright 2021 NVIDIA Corporation
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
from test_tools.generators import scalar_gen

import cunumeric as num


def test():
    test_values = [-0.5, 0.5]
    for a in test_values:
        for (la, na) in zip(scalar_gen(num, a), scalar_gen(np, a)):
            assert np.array_equal(num.negative(la), np.negative(na))
            assert np.array_equal(-la, -na)


if __name__ == "__main__":
    test()
