# Copyright 2023 NVIDIA Corporation
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
from utils.comparisons import allclose

import cunumeric as num


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", ([1, 5, 8], [1, 2, 3, 4, 5, 6, 7]))
def test_histogram_no_weights(src, bins):
    eps = 1.0e-8
    src_array = np.array(src)

    if num.isscalar(bins):
        bins_maybe_array = bins
    else:
        bins_maybe_array = np.array(bins)
        assert bins_maybe_array.shape[0] > 0

    np_out, np_bins_out = np.histogram(src_array, bins_maybe_array)
    num_out, num_bins_out = num.histogram(src_array, bins_maybe_array)

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", ([1, 5, 8], [1, 2, 3, 4, 5, 6, 7]))
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
@pytest.mark.parametrize("density", (False, True))
def test_histogram_weights(src, bins, weights, density):
    eps = 1.0e-8
    src_array = np.array(src)

    if num.isscalar(bins):
        bins_maybe_array = bins
    else:
        bins_maybe_array = np.array(bins)

    weights_array = np.array(weights)

    # density = False # remove me!

    np_out, np_bins_out = np.histogram(
        src_array, bins_maybe_array, weights=weights_array, density=density
    )
    num_out, num_bins_out = num.histogram(
        src_array, bins_maybe_array, weights=weights_array, density=density
    )

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", (5, 7))
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
@pytest.mark.parametrize("density", (False, True))
@pytest.mark.parametrize("ranges", ((3, 6), (1, 3)))
# @pytest.mark.skip(reason="debugging...")
def test_histogram_ranges(src, bins, weights, density, ranges):
    eps = 1.0e-8
    src_array = np.array(src)
    weights_array = np.array(weights)

    np_out, np_bins_out = np.histogram(
        src_array, bins, density=density, weights=weights_array, range=ranges
    )

    # print(("##### numpy: %s, %s") % (str(np_out), str(np_bins_out)))

    num_out, num_bins_out = num.histogram(
        src_array, bins, density=density, weights=weights_array, range=ranges
    )

    # print(("##### cunumeric: %s, %s") % (str(num_out), str(num_bins_out)))

    assert allclose(np_bins_out, num_bins_out, atol=eps)
    assert allclose(np_out, num_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([0, 2, 1, 3, 1, 4, 3, 2], [4, 2, 3, 3, 2, 4, 3, 2])
)
@pytest.mark.parametrize(
    "bins", ([5, 8, 14], [0, 0.1, 0.7, 1.0, 1.2], [1, 2, 3, 3, 5, 6, 7])
)
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
# @pytest.mark.skip(reason="debugging...")
def test_histogram_extreme_bins(src, bins, weights):
    eps = 1.0e-8
    src_array = np.array(src)
    bins_array = np.array(bins)
    weights_array = np.array(weights)

    np_out, np_bins_out = np.histogram(
        src_array, bins_array, weights=weights_array
    )
    num_out, num_bins_out = num.histogram(
        src_array, bins_array, weights=weights_array
    )

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([], [2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", (5, 7))
def test_histogram_no_weights_scalar_bin(src, bins):
    eps = 1.0e-8
    src_array = np.array(src)

    np_out, np_bins_out = np.histogram(src_array, bins)

    num_out, num_bins_out = num.histogram(src_array, bins)

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize(
    "src", ([2, 3, 3, 5, 2, 7, 6, 4], [0, 2, 1, 3, 1, 4, 3, 2])
)
@pytest.mark.parametrize("bins", (5, 7))
@pytest.mark.parametrize(
    "weights",
    (
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.1, 0.5, 0.1, 0.7, 0.2, 0.8, 1.3],
    ),
)
@pytest.mark.parametrize("density", (False, True))
def test_histogram_weights_scalar_bin(src, bins, weights, density):
    eps = 1.0e-8
    src_array = np.array(src)

    weights_array = np.array(weights)

    np_out, np_bins_out = np.histogram(
        src_array, bins, weights=weights_array, density=density
    )
    num_out, num_bins_out = num.histogram(
        src_array, bins, weights=weights_array, density=density
    )

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


@pytest.mark.parametrize("src", ([], [2], [5]))
@pytest.mark.parametrize("bins", ([1, 5, 8], [1, 2, 3, 4, 5, 6, 7]))
def test_histogram_singleton_empty(src, bins):
    eps = 1.0e-8
    src_array = np.array(src)

    bins_array = np.array(bins)

    np_out, np_bins_out = np.histogram(src_array, bins_array)

    num_out, num_bins_out = num.histogram(src_array, bins_array)

    assert allclose(np_out, num_out, atol=eps)
    assert allclose(np_bins_out, num_bins_out, atol=eps)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
