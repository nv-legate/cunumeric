# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See the LICENSE file for details.
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
