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

import numpy as np
import pytest

import cunumeric as num

anp = np.random.randn(4, 5)

@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [True, False])
def test_argmin(axis, keepdims):
    a = num.array(anp)

    assert np.array_equal(
        num.argmin(a, axis=axis, keepdims=keepdims),
        np.argmin(anp, axis=axis, keepdims=keepdims)
    )


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [True, False])
def test_argmax(axis, keepdims):
    a = num.array(anp)

    assert np.array_equal(
        num.argmax(a, axis=axis, keepdims=keepdims),
        np.argmax(anp, axis=axis, keepdims=keepdims)
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
