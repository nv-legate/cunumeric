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
    for dtype in [np.double, np.complex64]:
        np.random.seed(42)
        x_np = np.array(np.random.randn(11), dtype=dtype)
        y_np = np.array(np.random.randn(11), dtype=dtype)

        x_lg = lg.array(x_np)
        y_lg = lg.array(y_np)

        out_np = x_np.dot(y_np)
        out_lg = x_lg.dot(y_lg)

        assert lg.allclose(out_np, out_lg)


if __name__ == "__main__":
    test()
