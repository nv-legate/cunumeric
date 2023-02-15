# Copyright 2023 NVIDIA Corporation
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


# import numpy as np
import pytest

import cunumeric as num
import numpy as np


def my_func(a, b):
    a = a * 2 + b


def test_vectorize():
    func = num.vectorize(my_func)
    a = num.arange(5)
    b = num.ones((5,))
    func(a, b)
    assert(np.array_equal(a, [1,3,5,7,9]))

    a= num.arange(5)
    b=2
    func(a,b)
    assert(np.array_equal(a, [2,4,6,8,10]))
    

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
