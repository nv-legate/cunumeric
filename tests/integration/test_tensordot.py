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
import pytest
from cunumeric.utils import tensordot_modes
from test_tools.contractions import check_default

from legate.core import LEGATE_MAX_DIM


def gen_axes(a_ndim, b_ndim):
    yield from range(min(a_ndim, b_ndim, 2) + 1)
    if a_ndim >= 2 and b_ndim >= 2:
        yield ([0, 1], [0, 1])
        yield ([0, 1], [1, 0])


@pytest.mark.parametrize("b_ndim", range(LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("a_ndim", range(LEGATE_MAX_DIM + 1))
def test_tensordot(a_ndim, b_ndim):
    for axes in gen_axes(a_ndim, b_ndim):
        name = f"tensordot({a_ndim} x {b_ndim}, axes={axes})"
        modes = tensordot_modes(a_ndim, b_ndim, axes)

        def operation(lib, *args, **kwargs):
            return lib.tensordot(*args, **kwargs, axes=axes)

        check_default(name, modes, operation)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
