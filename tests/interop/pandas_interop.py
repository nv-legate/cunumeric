# Copyright 2017-2022 NVIDIA Corporation
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

import legate.pandas as pd

import cunumeric as np


def test():
    sr = pd.Series([1, 2, 3])
    array = np.asarray(sr)
    x = np.array([1, 2, 3])
    assert np.array_equal(array, x)

    y = np.array([4, 5, 6])
    z = np.add(sr, y)
    assert np.array_equal(z, x + y)

    df = pd.DataFrame({"x": x, "y": y})
    z = np.add(df["x"], df["y"])
    assert np.array_equal(z, x + y)
    return


if __name__ == "__main__":
    test()
