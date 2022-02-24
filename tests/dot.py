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

from cunumeric.utils import dot_modes
from test_tools.contractions import (
    test_default,
    test_permutations,
    test_shapes,
    test_types,
)

from legate.core import LEGATE_MAX_DIM


def test():
    for a_ndim in range(LEGATE_MAX_DIM + 1):
        for b_ndim in range(LEGATE_MAX_DIM + 1):
            name = f"dot({a_ndim} x {b_ndim})"
            modes = dot_modes(a_ndim, b_ndim)

            def operation(lib, *args, **kwargs):
                return lib.dot(*args, **kwargs)

            test_default(name, modes, operation)
            test_permutations(name, modes, operation)
            test_shapes(name, modes, operation)
            if a_ndim <= 2 and b_ndim <= 2:
                test_types(name, modes, operation)


if __name__ == "__main__":
    test()
