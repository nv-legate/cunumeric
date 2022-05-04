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
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from ..coverage import is_implemented, is_multi, is_single
from ._comparison_config import MISSING_NP_REFS, SKIP


@dataclass(frozen=True)
class ItemDetail:
    name: str
    implemented: bool
    np_ref: str
    lg_ref: str
    single: str
    multi: str


@dataclass(frozen=True)
class SectionDetail:
    title: str
    np_count: int
    lg_count: int
    items: list[ItemDetail]


def _npref(name, obj):
    if isinstance(obj, ModuleType):
        full_name = f"{obj.__name__}.{name}"
    else:
        full_name = f"numpy.{obj.__name__}.{name}"

    role = "meth" if "ndarray" in full_name else "obj"

    if full_name in MISSING_NP_REFS:
        return f"``{full_name}``"
    return f":{role}:`{full_name}`"


def _lgref(name, obj, implemented):
    if not implemented:
        return "-"

    if isinstance(obj, ModuleType):
        full_name = f"{obj.__name__}.{name}"
    else:
        full_name = f"cunumeric.{obj.__name__}.{name}"

    role = "meth" if "ndarray" in full_name else "obj"

    return f":{role}:`{full_name}`"


def filter_names(obj, types=None, use_skip=True):
    names = (n for n in dir(obj))  # every name in the module or class
    if use_skip:
        names = (n for n in names if n not in SKIP)  # except the ones we skip
    names = (n for n in names if not n.startswith("_"))  # or any private names
    if types:
        # optionally filtered by type
        names = (n for n in names if isinstance(getattr(obj, n), types))
    return names


def get_item(name, np_obj, lg_obj):
    lg_attr = getattr(lg_obj, name)

    implemented = is_implemented(lg_attr)

    if implemented:
        single = "YES" if is_single(lg_attr) else "NO"
        multi = "YES" if is_multi(lg_attr) else "NO"
    else:
        single = multi = ""

    return ItemDetail(
        name=name,
        implemented=implemented,
        np_ref=_npref(name, np_obj),
        lg_ref=_lgref(name, lg_obj, implemented),
        single=single,
        multi=multi,
    )


def get_namespaces(attr):
    import numpy

    import cunumeric

    if attr is None:
        return numpy, cunumeric

    return getattr(numpy, attr), getattr(cunumeric, attr)


def generate_section(config):
    np_obj, lg_obj = get_namespaces(config.attr)

    if config.names:
        names = config.names
    else:
        names = filter_names(np_obj, config.types)

    items = [get_item(name, np_obj, lg_obj) for name in names]

    return SectionDetail(
        title=config.title,
        np_count=len(items),
        lg_count=len([item for item in items if item.implemented]),
        items=sorted(items, key=lambda x: x.name),
    )
