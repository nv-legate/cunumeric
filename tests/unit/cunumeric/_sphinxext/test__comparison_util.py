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

import num._sphinxext._comparison_util as m  # module under test
import numpy as np
import pytest

import cunumeric as num


def test_get_namespaces_None():
    res = m.get_namespaces(None)
    assert len(res) == 2
    assert res[0] is np
    assert res[1] is num


@pytest.mark.parametrize("attr", ("ndarray", "linalg"))
def test_get_namespaces_attr(attr):
    res = m.get_namespaces(attr)
    assert len(res) == 2
    assert res[0] is getattr(np, attr)
    assert res[1] is getattr(num, attr)


class _TestObj:
    a = 10
    b = 10.2
    c = "str"
    _priv = "priv"


class Test_filter_names:
    def test_default(self):
        assert set(m.filter_names(_TestObj)) == {"a", "b", "c"}

    def test_types(self):
        assert set(m.filter_names(_TestObj, (int,))) == {"a"}
        assert set(m.filter_names(_TestObj, (int, str))) == {"a", "c"}
        assert set(m.filter_names(_TestObj, (int, set))) == {"a"}
        assert set(m.filter_names(_TestObj, (set,))) == set()

    def test_skip(self):
        assert set(m.filter_names(_TestObj, skip=("a",))) == {"b", "c"}
        assert set(m.filter_names(_TestObj, skip=("a", "c"))) == {"b"}


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
