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


def test_permutation_int():
    count = 1024
    p = num.random.permutation(count)
    p.sort()
    assert num.linalg.norm(p - np.arange(count)) == 0.0


def test_permutation_array():
    count = 1024
    x = num.arange(count)
    p = num.random.permutation(x)
    assert num.linalg.norm(x - p) != 0.0
    p.sort()
    assert num.linalg.norm(x - p) == 0.0


def test_shuffle():
    count = 16
    p = num.arange(count)
    x = num.arange(count)
    num.random.shuffle(x)
    assert num.linalg.norm(x - p) != 0.0
    x.sort()
    assert num.linalg.norm(x - p) == 0.0


class TestChoice:

    maxvalue = 1024
    count = 42

    def test_choice_1(self):
        a = num.random.choice(self.maxvalue, self.count)
        assert len(a) == self.count
        assert num.amax(a) <= self.maxvalue
        assert num.amin(a) >= 0

    def test_choice_2(self):
        a = num.random.choice(self.maxvalue, self.count, False)
        assert len(a) == self.count
        assert num.amax(a) <= self.maxvalue
        assert num.amin(a) >= 0
        for i in range(self.count):
            for j in range(self.count):
                if i == j:
                    continue
                assert a[i] != a[j]

    def test_choice_3(self):
        values = num.random.random_integers(0, self.maxvalue, self.maxvalue)

        a = num.random.choice(values, self.count)
        assert len(a) == self.count
        assert num.amax(a) <= num.amax(values)
        assert num.amin(a) >= num.amin(values)

    def test_choice_4(self):
        values = num.arange(self.maxvalue)

        a = num.random.choice(values, self.count, False)
        assert len(a) == self.count
        assert num.amax(a) <= num.amax(values)
        assert num.amin(a) >= num.amin(values)
        for i in range(self.count):
            for j in range(self.count):
                if i == j:
                    continue
                assert a[i] != a[j]

    def test_choice_5(self):
        values = num.arange(self.maxvalue)

        p = num.random.uniform(0, 1, self.maxvalue)
        p /= p.sum()

        a = num.random.choice(values, self.count, True, p)
        assert len(a) == self.count
        assert num.amax(a) <= num.amax(values)
        assert num.amin(a) >= num.amin(values)

    def test_choice_6(self):
        values = num.random.random_integers(0, self.maxvalue, self.maxvalue)

        p = num.random.uniform(0, 1, self.maxvalue)
        p /= p.sum()

        a = num.random.choice(values, self.count, True, p)
        assert len(a) == self.count
        assert num.amax(a) <= num.amax(values)
        assert num.amin(a) >= num.amin(values)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
