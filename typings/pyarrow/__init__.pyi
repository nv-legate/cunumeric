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

from typing import Any, Union

from .lib import (
    DataType,
    binary,
    bool_,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    string,
    uint8,
    uint16,
    uint32,
    uint64,
)

class Field:
    name: str
    type: DataType
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def with_name(self, name: str) -> Field: ...

def field(
    name: Union[str, bytes],
    type: DataType,
    nullable: bool = True,
    metadata: Any = None,
) -> Field: ...

class Schema:
    types: Any
    def field(self, i: Union[str, int]) -> Field: ...
    def get_all_field_indices(self, name: str) -> list[int]: ...
    def get_field_index(self, name: str) -> int: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Field: ...

def schema(fields: Any, metadata: Any = None) -> Schema: ...

class ExtensionType:
    def __init__(self, dtype: DataType, name: str) -> None: ...

class DictionaryType: ...
class ListType: ...
class MapType: ...
class StructType: ...
class UnionType: ...
class TimestampType: ...
class Time32Type: ...
class Time64Type: ...
class FixedSizeBinaryType: ...
class Decimal128Type: ...
class time32: ...
class time64: ...
class timestamp: ...
class date32: ...
class date64: ...
class large_binary: ...
class large_string: ...
class large_utf8: ...
class decimal128: ...
class large_list: ...
class struct: ...
class dictionary: ...
class null: ...
class utf8: ...
class list_: ...
class map_: ...

def from_numpy_dtype(dtype: Any) -> DataType: ...

__all__ = (
    "binary",
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "Field",
    "Schema",
    "DataType",
    "DictionaryType",
    "ListType",
    "MapType",
    "StructType",
    "UnionType",
    "TimestampType",
    "Time32Type",
    "Time64Type",
    "FixedSizeBinaryType",
    "Decimal128Type",
    "time32",
    "time64",
    "timestamp",
    "date32",
    "date64",
    "string",
    "large_binary",
    "large_string",
    "large_utf8",
    "decimal128",
    "large_list",
    "struct",
    "dictionary",
    "null",
    "utf8",
    "list_",
    "map_",
    "from_numpy_dtype",
)
