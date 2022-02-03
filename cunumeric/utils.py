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

import inspect
import traceback
import warnings
from functools import reduce

import numpy as np


# Get the list of attributes defined in a namespace
def getPredefinedAttributes(namespace):
    preDefined = {}
    for attr in dir(namespace):
        preDefined[attr] = getattr(namespace, attr)
    return preDefined


def find_last_user_stacklevel():
    stacklevel = 1
    for (frame, _) in traceback.walk_stack(None):
        if not frame.f_globals["__name__"].startswith("cunumeric"):
            break
        stacklevel += 1
    return stacklevel


def unimplemented(func):
    def wrapper(*args, **kwargs):
        find_last_user_stacklevel()

        warnings.warn(
            "cuNumeric has not implemented "
            + func.__name__
            + " and is falling back to canonical numpy. You may notice "
            + "significantly decreased performance for this function call.",
            stacklevel=2,
            category=RuntimeWarning,
        )
        return func(*args, **kwargs)

    return wrapper


# Copy attributes from one module to another.
# Works only on modules and doesnt add submodules
def add_missing_attributes(baseModule, definedModule):
    internal_attrs = set(["__dir__", "__getattr__"])
    preDefined = getPredefinedAttributes(definedModule)
    attrList = {}
    for attr in dir(baseModule):
        if attr in preDefined:
            pass
        elif not inspect.ismodule(getattr(baseModule, attr)):
            attrList[attr] = getattr(baseModule, attr)

    # add the attributes
    for key, value in attrList.items():
        if (
            callable(value)
            and not isinstance(value, type)
            and key not in internal_attrs
        ):
            setattr(definedModule, key, unimplemented(value))
        else:
            setattr(definedModule, key, value)


# These are the dtypes that we currently support for cuNumeric
def is_supported_dtype(dtype):
    assert isinstance(dtype, np.dtype)
    base_type = dtype.type
    if (
        base_type == np.float16
        or base_type == np.float32
        or base_type == np.float64
        or base_type == float
    ):
        return True
    if (
        base_type == np.int16
        or base_type == np.int32
        or base_type == np.int64
        or base_type == int
    ):
        return True
    if (
        base_type == np.uint16
        or base_type == np.uint32
        or base_type == np.uint64
    ):  # noqa E501
        return True
    if base_type == np.bool_ or base_type == bool:
        return True
    return False


def calculate_volume(shape):
    if shape == ():
        return 0
    return reduce(lambda x, y: x * y, shape)


def get_arg_dtype(dtype):
    return np.dtype(
        [("arg", np.int64), ("arg_value", dtype)],
        align=True,
    )


def get_arg_value_dtype(dtype):
    return dtype.fields["arg_value"][0].type
