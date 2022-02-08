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

import numpy as np

import cunumeric as num


def test():
    bases_np = np.random.randn(4, 5)

    # avoid fractional exponents
    exponents_np = np.random.randint(10, size=(4, 5)).astype(np.float64)

    bases = num.array(bases_np)
    exponents = num.array(exponents_np)

    assert num.allclose(
        num.power(bases, exponents), np.power(bases_np, exponents_np)
    )


if __name__ == "__main__":
    test()
