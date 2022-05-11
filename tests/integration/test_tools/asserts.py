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


def assert_equal(lgarr, nparr):
    if isinstance(lgarr, tuple) and isinstance(nparr, tuple):
        for resultnp, resultlg in zip(np.array(nparr), lgarr):
            assert np.array_equal(resultnp, resultlg)
    else:
        assert np.array_equal(lgarr, nparr)


def assert_allclose(lgarr, nparr, rtol=1e-05, atol=1e-08, equal_nan=False):
    if isinstance(lgarr, tuple) and isinstance(nparr, tuple):
        for resultnp, resultlg in zip(np.array(nparr), lgarr):
            assert np.allclose(
                resultnp, resultlg, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
    else:
        assert np.allclose(
            lgarr, nparr, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
