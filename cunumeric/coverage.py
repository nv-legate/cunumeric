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

import inspect
import warnings

from .runtime import runtime
from .utils import find_last_user_frames, find_last_user_stacklevel


# Get the list of attributes defined in a namespace
def getPredefinedAttributes(namespace):
    preDefined = {}
    for attr in dir(namespace):
        preDefined[attr] = getattr(namespace, attr)
    return preDefined


def unimplemented(func):
    def wrapper(*args, **kwargs):
        """Unimplemented"""
        stacklevel = find_last_user_stacklevel()

        warnings.warn(
            "cuNumeric has not implemented "
            + func.__name__
            + " and is falling back to canonical numpy. "
            + "You may notice significantly decreased performance "
            + "for this function call.",
            stacklevel=stacklevel,
            category=RuntimeWarning,
        )
        return func(*args, **kwargs)

    return wrapper


def unimplemented_with_reporting(func_name, func):
    def wrapper(*args, **kwargs):
        loc = find_last_user_frames(not runtime.report_dump_callstack)
        runtime.record_api_call(func_name, loc, False)
        return func(*args, **kwargs)

    return wrapper


def implemented(func_name, func):
    def wrapper(*args, **kwargs):
        loc = find_last_user_frames(not runtime.report_dump_callstack)
        runtime.record_api_call(func_name, loc, True)
        return func(*args, **kwargs)

    return wrapper


# Copy attributes from one module to another.
# Works only on modules and doesnt add submodules
def add_missing_attributes(baseModule, definedModule):
    module_name = baseModule.__name__
    internal_attrs = set(["__dir__", "__getattr__"])
    preDefined = getPredefinedAttributes(definedModule)
    attrList = {}
    for attr in dir(baseModule):
        if attr in preDefined:
            pass
        elif not inspect.ismodule(getattr(baseModule, attr)):
            attrList[attr] = getattr(baseModule, attr)

    if runtime.report_coverage:
        for key, value in preDefined.items():
            if callable(value):
                wrapped = implemented(f"{module_name}.{key}", value)
                setattr(definedModule, key, wrapped)

    # add the attributes
    for key, value in attrList.items():
        if (
            callable(value)
            and not isinstance(value, type)
            and key not in internal_attrs
        ):
            if runtime.report_coverage:
                wrapped = unimplemented_with_reporting(
                    f"{module_name}.{key}", value
                )
                setattr(definedModule, key, wrapped)
            else:
                setattr(definedModule, key, unimplemented(value))
        else:
            setattr(definedModule, key, value)
