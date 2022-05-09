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

from sphinx.ext.autodoc import FunctionDocumenter

from cunumeric import ufunc


class UfuncDocumenter(FunctionDocumenter):
    directivetype = "function"
    objtype = "ufunc"
    priority = 20

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(getattr(member, "__wrapped__", None), ufunc)


def setup(app):
    app.add_autodocumenter(UfuncDocumenter)
