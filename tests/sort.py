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
    np.random.seed(42)
    A_np = np.array(np.random.randint(10, size=30), dtype=np.int32)

    A_num = num.array(A_np)
    print("Sorting array   : " + str(A_np))

    sortA_np = np.sort(A_np)
    print("Result numpy    : " + str(sortA_np))

    # pdb.set_trace()
    sortA_num = num.sort(A_num)
    print("Result cunumeric: " + str(sortA_num))
    assert num.allclose(sortA_np, sortA_num)

    A_num.sort()
    print("Result (inplace): " + str(A_num))
    assert num.allclose(sortA_np, A_num)

    return


if __name__ == "__main__":
    test()
