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

import numpy as np

import cunumeric as lg


def test():
    anp = np.random.randn(10)
    a = lg.array(anp)

    assert np.allclose(np.linalg.norm(anp), lg.linalg.norm(a))
    assert np.allclose(
        np.linalg.norm(anp, ord=np.inf), lg.linalg.norm(a, ord=lg.inf)
    )
    assert np.allclose(
        np.linalg.norm(anp, ord=-np.inf), lg.linalg.norm(a, ord=-lg.inf)
    )
    assert np.allclose(np.linalg.norm(anp, ord=0), lg.linalg.norm(a, ord=0))
    assert np.allclose(np.linalg.norm(anp, ord=1), lg.linalg.norm(a, ord=1))
    # assert(np.allclose(np.linalg.norm(anp, ord=-1), lg.linalg.norm(a, ord=-1))) # noqa E501
    assert np.allclose(np.linalg.norm(anp, ord=2), lg.linalg.norm(a, ord=2))
    # assert(np.allclose(np.linalg.norm(anp, ord=-2), lg.linalg.norm(a, ord=-2))) # noqa E501

    # bnp = np.random.randn(4,5)
    # b = lg.array(bnp)
    # assert(np.allclose(np.linalg.norm(bnp, 'fro'), lg.linalg.norm(b, 'fro')))
    # assert(np.allclose(np.linalg.norm(bnp, 'nuc'), lg.linalg.norm(b, 'nuc')))
    # assert(np.allclose(np.linalg.norm(bnp, np.inf), lg.linalg.norm(b, lg.inf))) # noqa E501
    # assert(np.allclose(np.linalg.norm(bnp, -np.inf), lg.linalg.norm(b, -lg.inf))) # noqa E501
    # assert(np.allclose(np.linalg.norm(bnp, 1), lg.linalg.norm(b, 1)))
    # assert(np.allclose(np.linalg.norm(bnp, -1), lg.linalg.norm(b, -1)))

    return


if __name__ == "__main__":
    test()
