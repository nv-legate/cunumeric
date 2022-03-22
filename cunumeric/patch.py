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
""" This module may be imported in order to globably replace NumPy with
CuNumeric.

In order to function properly, this module must be imported early (ideally
at the very start of a script).  The ``numpy`` module in ``sys.modules``
will be replaced with ``cunumeric`` so that any subsequent use of the
``numpy`` module will use ``cunumeric`` instead.

This module is primarily intended for quick demonstrations or proofs of
concept.

"""

import sys

import cunumeric

sys.modules["numpy"] = cunumeric
