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
from __future__ import annotations

from typing import Literal, Tuple, Union

from typing_extensions import TypeAlias

NdShape: TypeAlias = Tuple[int, ...]

NdShapeLike: TypeAlias = Union[int, NdShape]

SortSide: TypeAlias = Literal["left", "right"]

SortType: TypeAlias = Literal["quicksort", "mergesort", "heapsort", "stable"]

OrderType: TypeAlias = Literal["A", "C", "F"]

BitOrder: TypeAlias = Literal["big", "little"]

ConvolveMode: TypeAlias = Literal["full", "valid", "same"]

SelectKind: TypeAlias = Literal["introselect"]
