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

import legate.numpy as lg


def test():
    for k in [0, -1, 1, -2, 2]:
        print(f"np.eye(5, k={k})")
        e_lg = lg.eye(5, k=k)
        e_np = np.eye(5, k=k)
        assert np.array_equal(e_lg, e_np)

        print(f"np.eye(5, 6, k={k})")
        e_lg = lg.eye(5, 6, k=k)
        e_np = np.eye(5, 6, k=k)
        assert np.array_equal(e_lg, e_np)

        print(f"np.eye(5, 4, k={k})")
        e_lg = lg.eye(5, 4, k=k)
        e_np = np.eye(5, 4, k=k)
        assert np.array_equal(e_lg, e_np)

    return


if __name__ == "__main__":
    test()
