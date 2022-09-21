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

import warnings
from dataclasses import dataclass
from functools import wraps
from types import FunctionType, MethodDescriptorType, MethodType, ModuleType
from typing import Any, Container, Mapping, Optional, cast

import numpy as np
from typing_extensions import Protocol

from legate.core import track_provenance

from .runtime import runtime
from .utils import find_last_user_frames, find_last_user_stacklevel

__all__ = ("clone_module", "clone_np_ndarray")

FALLBACK_WARNING = (
    "cuNumeric has not implemented {name} "
    + "and is falling back to canonical numpy. "
    + "You may notice significantly decreased performance "
    + "for this function call."
)

MOD_INTERNAL = {"__dir__", "__getattr__"}

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


def filter_namespace(
    ns: Mapping[str, Any],
    *,
    omit_names: Optional[Container[str]] = None,
    omit_types: tuple[type, ...] = (),
) -> dict[str, Any]:
    omit_names = omit_names or set()
    return {
        attr: value
        for attr, value in ns.items()
        if attr not in omit_names and not isinstance(value, omit_types)
    }


class AnyCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class CuWrapperMetadata:
    implemented: bool
    single: bool = False
    multi: bool = False


class CuWrapped(AnyCallable, Protocol):
    _cunumeric: CuWrapperMetadata
    __wrapped__: Any
    __name__: str
    __qualname__: str


def implemented(
    func: AnyCallable, prefix: str, name: str, reporting: bool = True
) -> CuWrapped:
    name = f"{prefix}.{name}"

    wrapper: CuWrapped

    if reporting:

        @wraps(func)
        @track_provenance(runtime.legate_context, nested=True)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            location = find_last_user_frames(
                not runtime.args.report_dump_callstack
            )
            runtime.record_api_call(
                name=name,
                location=location,
                implemented=True,
            )
            return func(*args, **kwargs)

    else:

        @wraps(func)
        @track_provenance(runtime.legate_context, nested=True)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    # This is incredibly ugly and unpleasant, but @wraps(func) doesn't handle
    # ufuncs the way we need it to. The alternative would be to vendor and
    # modify a custom version of @wraps
    if hasattr(wrapper.__wrapped__, "_name"):
        wrapper.__name__ = wrapper.__wrapped__._name
        wrapper.__qualname__ = wrapper.__wrapped__._name

    # TODO (bev) Scraping text to set flags seems a bit fragile. It would be
    # preferable to start with flags, and use those to update docstrings.
    multi = "Multiple GPUs" in (getattr(func, "__doc__", None) or "")
    single = "Single GPU" in (getattr(func, "__doc__", None) or "") or multi

    wrapper._cunumeric = CuWrapperMetadata(
        implemented=True, single=single, multi=multi
    )

    return wrapper


def unimplemented(
    func: AnyCallable,
    prefix: str,
    name: str,
    reporting: bool = True,
    self_fallback: Optional[str] = None,
) -> CuWrapped:
    name = f"{prefix}.{name}"

    # Skip over NumPy's `__array_function__` dispatch wrapper, if present.
    # NumPy adds `__array_function__` dispatch logic through decorators, but
    # still makes the underlying code (which converts all array-like arguments
    # to `numpy.ndarray` through `__array__`) available in the
    # `_implementation` field.
    # We have to skip the dispatch wrapper, otherwise we will trigger an
    # infinite loop. Say we're dealing with a call to `cunumeric.foo`, and are
    # trying to fall back to `numpy.foo`. If we didn't skip the dispatch
    # wrapper of `numpy.foo`, then NumPy would ask
    # `cunumeric.ndarray.__array_function__` to handle the call to `numpy.foo`,
    # then `cunumeric.ndarray.__array_function__` would call `cunumeric.foo`,
    # and we would end up here again.
    func = getattr(func, "_implementation", func)

    wrapper: CuWrapped

    if reporting:

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            location = find_last_user_frames(
                not runtime.args.report_dump_callstack
            )
            runtime.record_api_call(
                name=name,
                location=location,
                implemented=False,
            )
            if self_fallback:
                self_value = getattr(args[0], self_fallback)()
                args = (self_value,) + args[1:]
            return func(*args, **kwargs)

    else:

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            stacklevel = find_last_user_stacklevel()
            warnings.warn(
                FALLBACK_WARNING.format(name=name),
                stacklevel=stacklevel,
                category=RuntimeWarning,
            )
            if self_fallback:
                self_value = getattr(args[0], self_fallback)()
                args = (self_value,) + args[1:]
            return func(*args, **kwargs)

    wrapper._cunumeric = CuWrapperMetadata(implemented=False)

    return wrapper


def clone_module(
    origin_module: ModuleType, new_globals: dict[str, Any]
) -> None:
    """Copy attributes from one module to another, excluding submodules

    Function types are wrapped with a decorator to report API calls. All
    other values are copied as-is.

    Parameters
    ----------
    origin_module : ModuleTpe
        Existing module to clone attributes from

    new_globals : dict
        a globals() dict for the new module to clone into

    Returns
    -------
    None

    """
    mod_name = origin_module.__name__

    missing = filter_namespace(
        origin_module.__dict__,
        omit_names=set(new_globals).union(MOD_INTERNAL),
        omit_types=(ModuleType,),
    )

    reporting = runtime.args.report_coverage

    from ._ufunc.ufunc import ufunc as lgufunc

    for attr, value in new_globals.items():
        # Only need to wrap things that are in the origin module to begin with
        if attr not in origin_module.__dict__:
            continue
        if isinstance(value, (FunctionType, lgufunc)):
            wrapped = implemented(
                cast(AnyCallable, value), mod_name, attr, reporting=reporting
            )
            new_globals[attr] = wrapped

    from numpy import ufunc as npufunc

    for attr, value in missing.items():
        if isinstance(value, (FunctionType, npufunc)):
            wrapped = unimplemented(value, mod_name, attr, reporting=reporting)
            new_globals[attr] = wrapped
        else:
            new_globals[attr] = value


def should_wrap(obj: object) -> bool:
    return isinstance(obj, (FunctionType, MethodType, MethodDescriptorType))


def clone_np_ndarray(cls: type) -> type:
    """Copy attributes from np.ndarray to cunumeric.ndarray

    Method types are wrapped with a decorator to report API calls. All
    other values are copied as-is.

    """

    origin_class = np.ndarray

    class_name = f"{origin_class.__module__}.{origin_class.__name__}"

    missing = filter_namespace(
        origin_class.__dict__,
        # this simply omits ndarray internal methods for any class. If
        # we ever need to wrap more classes we may need to generalize to
        # per-class specification of internal names to skip
        omit_names=set(cls.__dict__).union(NDARRAY_INTERNAL),
    )

    reporting = runtime.args.report_coverage

    for attr, value in cls.__dict__.items():
        # Only need to wrap things that are in the origin class to begin with
        if not hasattr(origin_class, attr):
            continue
        if should_wrap(value):
            wrapped = implemented(value, class_name, attr, reporting=reporting)
            setattr(cls, attr, wrapped)

    for attr, value in missing.items():
        if should_wrap(value):
            wrapped = unimplemented(
                value,
                class_name,
                attr,
                reporting=reporting,
                self_fallback="__array__",
            )
            setattr(cls, attr, wrapped)
        else:
            setattr(cls, attr, value)

    return cls


def is_implemented(obj: Any) -> bool:
    return hasattr(obj, "_cunumeric") and obj._cunumeric.implemented


def is_single(obj: Any) -> bool:
    return hasattr(obj, "_cunumeric") and obj._cunumeric.single


def is_multi(obj: Any) -> bool:
    return hasattr(obj, "_cunumeric") and obj._cunumeric.multi
