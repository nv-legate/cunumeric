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

from legate.core import LEGATE_MAX_DIM


def scalar_gen(lib, val):
    # pure scalar values
    yield lib.array(val)
    # ()-shape arrays
    yield lib.full((), val)
    for ndim in range(1, LEGATE_MAX_DIM):  # off-by-one is by design
        # singleton arrays
        yield lib.full(ndim * (1,), val)
        # singleton slices of larger arrays
        # TODO: disabled; currently core can't handle unary/binary operations
        # with future-backed output but regionfield-backed inputs
        # yield lib.full(ndim * (5,), val)[ndim * (slice(1, 2),)]
