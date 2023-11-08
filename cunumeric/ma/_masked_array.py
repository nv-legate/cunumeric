# Copyright 2023 NVIDIA Corporation
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
from __future__ import annotations

import numpy as _np

from ..array import maybe_convert_to_np_ndarray
from ..coverage import clone_class

NDARRAY_INTERNAL = {
    "__array_finalize__",
    "__array_function__",
    "__array_interface__",
    "__array_prepare__",
    "__array_priority__",
    "__array_struct__",
    "__array_ufunc__",
    "__array_wrap__",
}

MaskType = _np.bool_
nomask = MaskType(0)


@clone_class(_np.ma.MaskedArray, NDARRAY_INTERNAL, maybe_convert_to_np_ndarray)
class MaskedArray:
    def __new__(cls, *args, **kw):
        return super().__new__(cls)

    def __init__(
        self,
        data=None,
        mask=nomask,
        dtype=None,
        copy=False,
        subok=True,
        ndmin=0,
        fill_value=None,
        keep_mask=True,
        hard_mask=None,
        shrink=True,
        order=None,
    ):
        self._internal_ma = _np.ma.MaskedArray(
            data=maybe_convert_to_np_ndarray(data),
            mask=maybe_convert_to_np_ndarray(mask),
            dtype=dtype,
            copy=copy,
            subok=subok,
            ndmin=ndmin,
            fill_value=fill_value,
            keep_mask=keep_mask,
            hard_mask=hard_mask,
            shrink=shrink,
            order=order,
        )

    def __array__(self, _dtype=None):
        return self._internal_ma

    @property
    def size(self):
        return self._internal_ma.size

    @property
    def shape(self):
        return self._internal_ma.shape
