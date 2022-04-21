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

import cunumeric as num


def test():
    pythonX = [1, 2, 3]
    legate_oldX = num.array(pythonX)

    assert len(legate_oldX) == len(pythonX)

    SIZE = 100
    legate_oldX = num.random.random(SIZE)
    assert SIZE == len(legate_oldX)

    x = num.array([1, 2, 3, 4])
    y = num.array([1, 2, 3, 4])

    z = x + y

    print(len(z))
    assert len(x) == len(y) == len(z)

    x = num.array(pythonX)
    x = num.sqrt(x)

    assert len(x) == len(pythonX)


if __name__ == "__main__":
    test()
