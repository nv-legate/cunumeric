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

KS = [0, -1, 1, -2, 2]


def _test(func, k):
    num_f = getattr(num, func)
    np_f = getattr(np, func)

    a = num_f(100, k=k)
    an = np_f(100, k=k)
    assert num.array_equal(a, an)

    a = num_f(100, k=k, m=30)
    an = np_f(100, k=k, m=30)
    assert num.array_equal(a, an)


def _test_from(func, k):
    num_f = getattr(num, func)
    np_f = getattr(np, func)
    a = num.ones((70, 40), dtype=int)
    an = np.ones((70, 40), dtype=int)

    b = num_f(a, k=k)
    bn = np_f(an, k=k)
    assert num.array_equal(b, bn)


@pytest.mark.parametrize("k", KS, ids=lambda k: f"(k={k})")
def test_tril_indices_from(k):
    _test_from("tril_indices_from", k)


@pytest.mark.parametrize("k", KS, ids=lambda k: f"(k={k})")
def test_triu_indices_from(k):
    _test_from("triu_indices_from", k)


@pytest.mark.parametrize("k", KS, ids=lambda k: f"(k={k})")
def test_tril_indices(k):
    _test("tril_indices", k)


@pytest.mark.parametrize("k", KS, ids=lambda k: f"(k={k})")
def test_triu_indices(k):
    _test("triu_indices", k)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
