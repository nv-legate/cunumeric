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

import re

regex = re.compile(r"[\s]*Examples[^\n]*")


def _cut_out_examples(doc):
    return doc.split("Examples")[0].rstrip()


def copy_docstring(other):
    # Cut out the examples section and on from each docstring,
    # as it is not quite applicable to Legate Pandas.
    doc = _cut_out_examples(other.__doc__)

    def wrapper(obj):
        if callable(obj):
            obj.__doc__ = doc
        else:
            obj = property(obj.fget, obj.fset, obj.fdel, doc)
        return obj

    return wrapper
