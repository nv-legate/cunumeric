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


# import numpy as np
import pytest

import cunumeric as num


def my_func(a, b):
    a = a * 2 + b
    a = a * 3


def test_vectorize():
    func = num.vectorize(my_func)
    a = num.arange(5)
    b = num.zeros((5,))
    # b = 2
    func(a, b)
    # assert(a==12)
    print("IRINA DEBUG:")
    print(a)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
