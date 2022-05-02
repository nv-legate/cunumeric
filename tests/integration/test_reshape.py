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

SQUARE_CASES = [
    (10, 5, 2),
    (5, 2, 10),
    (5, 2, 5, 2),
    (10, 10, 1),
    (10, 1, 10),
    (1, 10, 10),
]


class TestSquare:

    anp = np.arange(100).reshape(10, 10)

    def test_basic(self):
        a = num.arange(100).reshape((10, 10))
        assert np.array_equal(self.anp, a)

    @pytest.mark.parametrize("shape", SQUARE_CASES, ids=str)
    def test_shape(self, shape):
        a = num.arange(100).reshape((10, 10))
        assert np.array_equal(
            num.reshape(a, shape),
            np.reshape(self.anp, shape),
        )

    def test_1d(self):
        a = num.arange(100).reshape((10, 10))
        assert np.array_equal(
            num.reshape(a, (100,)),
            np.reshape(self.anp, (100,)),
        )

    def test_ravel(self):
        a = num.arange(100).reshape((10, 10))
        assert np.array_equal(
            num.ravel(a),
            np.ravel(self.anp),
        )


RECT_CASES = [
    (10, 2, 10),
    (20, 10),
    (5, 40),
    (200, 1),
    (1, 200),
    (10, 20),
]


class TestRect:

    anp = np.random.rand(5, 4, 10)

    @pytest.mark.parametrize("shape", RECT_CASES, ids=str)
    def test_shape(self, shape):
        a = num.array(self.anp)
        assert np.array_equal(
            num.reshape(a, shape),
            np.reshape(self.anp, shape),
        )

    def test_1d(self):
        a = num.array(self.anp)
        assert np.array_equal(
            num.reshape(a, (200,)),
            np.reshape(self.anp, (200,)),
        )

    def test_ravel(self):
        a = num.array(self.anp)
        assert np.array_equal(
            num.ravel(a),
            np.ravel(self.anp),
        )


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
