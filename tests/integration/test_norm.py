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
import pytest

import cunumeric as num


def test():
    anp = np.random.randn(10)
    a = num.array(anp)

    assert np.allclose(np.linalg.norm(anp), num.linalg.norm(a))
    assert np.allclose(
        np.linalg.norm(anp, ord=np.inf), num.linalg.norm(a, ord=num.inf)
    )
    assert np.allclose(
        np.linalg.norm(anp, ord=-np.inf), num.linalg.norm(a, ord=-num.inf)
    )
    assert np.allclose(np.linalg.norm(anp, ord=0), num.linalg.norm(a, ord=0))
    assert np.allclose(np.linalg.norm(anp, ord=1), num.linalg.norm(a, ord=1))
    # assert(np.allclose(np.linalg.norm(anp, ord=-1), num.linalg.norm(a, ord=-1))) # noqa E501
    assert np.allclose(np.linalg.norm(anp, ord=2), num.linalg.norm(a, ord=2))
    # assert(np.allclose(np.linalg.norm(anp, ord=-2), num.linalg.norm(a, ord=-2))) # noqa E501

    # bnp = np.random.randn(4,5)
    # b = num.array(bnp)
    # assert(np.allclose(np.linalg.norm(bnp, 'fro'), num.linalg.norm(b, 'fro'))) # noqa E501
    # assert(np.allclose(np.linalg.norm(bnp, 'nuc'), num.linalg.norm(b, 'nuc'))) # noqa E501
    # assert(np.allclose(np.linalg.norm(bnp, np.inf), num.linalg.norm(b, num.inf))) # noqa E501
    # assert(np.allclose(np.linalg.norm(bnp, -np.inf), num.linalg.norm(b, -num.inf))) # noqa E501
    # assert(np.allclose(np.linalg.norm(bnp, 1), num.linalg.norm(b, 1)))
    # assert(np.allclose(np.linalg.norm(bnp, -1), num.linalg.norm(b, -1)))


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
