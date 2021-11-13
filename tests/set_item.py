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

import cunumeric as num


def test():
    x = num.array([0, 0, 0])
    x[0] = 1
    x[1] = 2
    x[2] = 3

    assert x[0] == 1
    assert x[1] == 2
    assert x[2] == 3

    return


if __name__ == "__main__":
    test()
