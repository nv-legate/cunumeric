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


import numpy as np
import pytest

import cunumeric as cu


@pytest.mark.parametrize(
    "str_method",
    (
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "linear",
        "median_unbiased",
        "normal_unbiased",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ),
)
@pytest.mark.parametrize("axes", (0, 1, (0, 2)))
@pytest.mark.parametrize(
    "qin_arr", (0.5, [0.001, 0.37, 0.42, 0.67, 0.83, 0.99, 0.39, 0.49, 0.5])
)
@pytest.mark.parametrize("keepdims", (False, True))
def test_quantiles_1(str_method, axes, qin_arr, keepdims):
    eps = 1.0e-8
    arr = np.ndarray(
        shape=(2, 3, 4),
        buffer=np.array(
            [
                1,
                2,
                2,
                40,
                1,
                1,
                2,
                1,
                0,
                10,
                3,
                3,
                40,
                15,
                3,
                7,
                5,
                4,
                7,
                3,
                5,
                1,
                0,
                9,
            ]
        ),
        dtype=int,
    )

    if cu.isscalar(qin_arr):
        qs_arr = qin_arr
    else:
        qs_arr = np.array(qin_arr)

    # cunumeric:
    # print("cunumeric axis = %d:"%(axis))
    q_out = cu.quantile(
        arr, qs_arr, axis=axes, method=str_method, keepdims=keepdims
    )
    # print(q_out)

    # np:
    # print("numpy axis = %d:"%(axis))
    np_q_out = np.quantile(
        arr, qs_arr, axis=axes, method=str_method, keepdims=keepdims
    )
    # print(np_q_out)

    assert cu.all(q_out.shape == np_q_out.shape)
    assert q_out.dtype == np_q_out.dtype
    
    qo_flat = q_out.flatten().astype(np.float64)
    np_qo_flat = np_q_out.flatten().astype(np.float64)
    sz = qo_flat.size
    assert cu.all(
        [cu.abs(qo_flat[i] - np_qo_flat[i]) < eps for i in range(0, sz)]
    )


@pytest.mark.parametrize(
    "str_method",
    (
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "linear",
        "median_unbiased",
        "normal_unbiased",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ),
)
@pytest.mark.parametrize(
    "ls_in",
    (
        [[1.0, 0.13, 2.11], [1.9, 9.2, 0.17]],
        [
            [1, 1, 0],
            [2, 1, 10],
            [2, 2, 3],
            [40, 1, 3],
            [40, 5, 5],
            [15, 4, 1],
            [3, 7, 0],
            [7, 3, 9],
        ],
    ),
)
@pytest.mark.parametrize("axes", (0, 1))
@pytest.mark.parametrize("keepdims", (False, True))
def test_quantiles_2(str_method, ls_in, axes, keepdims):
    eps = 1.0e-8

    arr = np.array(ls_in)

    qs_arr = np.ndarray(
        shape=(2, 4),
        buffer=np.array([0.001, 0.37, 0.42, 0.5, 0.67, 0.83, 0.99, 0.39]).data,
    )

    # cunumeric:
    # print("cunumeric axis = %d:"%(axis))
    q_out = cu.quantile(
        arr, qs_arr, axis=axes, method=str_method, keepdims=keepdims
    )
    # print(q_out)

    # np:
    # print("numpy axis = %d:"%(axis))
    np_q_out = np.quantile(
        arr, qs_arr, axis=axes, method=str_method, keepdims=keepdims
    )
    # print(np_q_out)

    assert cu.all(q_out.shape == np_q_out.shape)
    assert q_out.dtype == np_q_out.dtype

    qo_flat = q_out.flatten().astype(np.float64)
    np_qo_flat = np_q_out.flatten().astype(np.float64)
    sz = qo_flat.size
    assert cu.all(
        [cu.abs(qo_flat[i] - np_qo_flat[i]) < eps for i in range(0, sz)]
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
