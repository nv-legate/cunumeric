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

import cunumeric as num


def test():
    anp = np.array([2 + 3j, 4 + 5j])
    a = num.array(anp)
    assert num.all(num.abs(num.exp2(a) - np.exp2(anp)) < 1e-5)
    return


if __name__ == "__main__":
    test()
