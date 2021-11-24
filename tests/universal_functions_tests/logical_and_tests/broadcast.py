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
    sample_arr = [True, False]
    anp = np.random.choice(sample_arr, size=10)
    a = num.array(anp)
    b = True

    # test logical_and on arrays different szes
    assert np.array_equal(num.logical_and(a, b), np.logical_and(anp, b))

    # test logical_and with scalar
    assert np.array_equal(num.logical_and(b, a), np.logical_and(b, anp))

    # operator interface
    assert np.array_equal(a & b, anp & b)
    assert np.array_equal(b & a, b & anp)

    return


if __name__ == "__main__":
    test()
