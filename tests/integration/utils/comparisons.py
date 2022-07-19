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

from itertools import islice
from typing import Any, Union

import numpy as np


def allclose(
    a: Any,  # numpy or cunumeric array-like
    b: Any,  # numpy or cunumeric array-like
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    *,
    diff_limit: Union[int, None] = 5,  # None means no limit at all
) -> bool:
    # simplify handling of scalar values
    a, b = np.atleast_1d(a), np.atleast_1d(b)

    close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    all_close = np.all(close)

    if (diff_limit is None or diff_limit > 0) and not all_close:
        a += np.zeros(b.shape, dtype=a.dtype)
        b += np.zeros(a.shape, dtype=b.dtype)
        inds = islice(zip(*np.where(not close)), diff_limit)
        diffs = [f"  index {i}: {a[i]} {b[i]}" for i in inds]
        N = len(diffs)
        print(
            f"First {N} difference{'s' if N>1 else ''} for allclose (with diff_limit={diff_limit}):\n"  # noqa E501
        )
        print("\n".join(diffs))

    return all_close
