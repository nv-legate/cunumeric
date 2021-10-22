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

from __future__ import division

import numpy as np

import cunumeric as lg

types = [np.complex64, np.complex128]


def test():
    for ty in types:
        x_np = np.array([1 + 4j, 2 + 5j, 3 + 6j], ty)
        x_lg = lg.array(x_np)

        assert lg.array_equal(x_np.conj(), x_lg.conj())
        assert lg.array_equal(x_np.real, x_lg.real)
        assert lg.array_equal(x_np.imag, x_lg.imag)

        x_np = np.array([3 + 6j], ty)
        x_lg = lg.array(x_np)

        assert lg.array_equal(x_np.conj(), x_lg.conj())
        assert lg.array_equal(x_np.real, x_lg.real)
        assert lg.array_equal(x_np.imag, x_lg.imag)


if __name__ == "__main__":
    test()
