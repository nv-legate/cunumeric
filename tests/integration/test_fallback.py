# Copyright 2022 NVIDIA Corporation
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
import pytest

import cunumeric as num


def test_array_equal():
    a = num.array([1, 2, 3])
    assert np.array_equal(a, a, equal_nan=True)


def test_ufunc():
    in_num = num.array([0, 1, 2, 3])
    in_np = in_num.__array__()

    # This test uses logical_xor.reduce because it is currently unimplemented,
    # and we want to verify a behaviour of unimplemented ufunc methods. If
    # logical_xor.reduce becomes implemented in the future, this assertion will
    # start to fail, and a new (unimplemented) ufunc method should be found to
    # replace it
    assert not num.logical_and.accumulate._cunumeric.implemented

    out_num = np.logical_and.accumulate(in_num)
    out_np = np.logical_and.accumulate(in_np)
    assert np.array_equal(out_num, out_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
