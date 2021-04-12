# Copyright 2021 NVIDIA Corporation
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

from legate.numpy.array import ndarray, runtime
from legate.numpy.config import NumPyOpCode
from legate.numpy.module import power as _power, sqrt as _sqrt


def norm(x, ord=None, axis=None, keepdims=False, stacklevel=1):
    lg_array = ndarray.convert_to_legate_ndarray(x)
    # Easy case to handle
    if lg_array.size == 1:
        return lg_array
    if (axis is None and lg_array.ndim == 1) or type(axis) == int:
        # Handle the weird norm cases
        if ord == np.inf:
            return abs(lg_array).max(
                axis=axis, keepdims=keepdims, stacklevel=(stacklevel + 1)
            )
        elif ord == -np.inf:
            return abs(lg_array).min(
                axis=axis, keepdims=keepdims, stacklevel=(stacklevel + 1)
            )
        elif ord == 0:
            # Check for where things are not zero and convert to integer
            # for sum
            temp = (lg_array != 0).astype(np.int64)
            return temp.sum(
                axis=axis, keepdims=keepdims, stacklevel=(stacklevel + 1)
            )
        elif ord is None:
            args = (np.array(2, dtype=np.int32),)
        elif type(ord) == int:
            args = (np.array(ord, dtype=np.int32),)
        else:
            raise ValueError("Invalid 'ord' argument passed to norm")
        result = ndarray.perform_unary_reduction(
            NumPyOpCode.NORM,
            NumPyOpCode.SUM_RADIX,
            lg_array,
            axis=axis,
            keepdims=keepdims,
            args=args,
            stacklevel=(stacklevel + 1),
        )
        # Now we need to do the power part if this is not an eager array
        # Eager arrays have the power part folded into their operation
        # TODO: fix this as it is a hack
        if (
            runtime.is_eager_array(result._thunk)
            and result._thunk.deferred is None
        ) or ord == 1:
            return result
        elif ord == 2 or ord is None:
            return _sqrt(result, stacklevel=stacklevel + 1)
        else:
            return _power(result, (1.0 / ord), stacklevel=(stacklevel + 1))
    else:
        raise NotImplementedError(
            "Legate needs support for other kinds of norms"
        )
