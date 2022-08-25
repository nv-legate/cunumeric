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


def test_choice_1(maxvalue=1024, count=42):
    a = num.random.choice(maxvalue, count)
    assert len(a) == count
    assert num.amax(a) <= maxvalue
    assert num.amin(a) >= 0


def test_choice_2(maxvalue=1024, count=42):
    a = num.random.choice(maxvalue, count, False)
    assert len(a) == count
    assert num.amax(a) <= maxvalue
    assert num.amin(a) >= 0
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            assert a[i] != a[j]


def test_choice_3(maxvalue=1024, count=42):
    values = num.random.random_integers(0, maxvalue, maxvalue)

    a = num.random.choice(values, count)
    assert len(a) == count
    assert num.amax(a) <= num.amax(values)
    assert num.amin(a) >= num.amin(values)


def test_choice_4(maxvalue=1024, count=42):
    values = num.arange(maxvalue)

    a = num.random.choice(values, count, False)
    assert len(a) == count
    assert num.amax(a) <= num.amax(values)
    assert num.amin(a) >= num.amin(values)
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            assert a[i] != a[j]


def test_choice_5(maxvalue=1024, count=42):
    values = num.arange(maxvalue)

    p = num.random.uniform(0, 1, maxvalue)
    p /= p.sum()

    a = num.random.choice(values, count, True, p)
    assert len(a) == count
    assert num.amax(a) <= num.amax(values)
    assert num.amin(a) >= num.amin(values)


def test_choice_6(maxvalue=1024, count=42):
    values = num.random.random_integers(0, maxvalue, maxvalue)

    p = num.random.uniform(0, 1, maxvalue)
    p /= p.sum()

    a = num.random.choice(values, count, True, p)
    assert len(a) == count
    assert num.amax(a) <= num.amax(values)
    assert num.amin(a) >= num.amin(values)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
