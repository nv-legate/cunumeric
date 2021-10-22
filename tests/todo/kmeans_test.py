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
    np.random.seed(50)
    datanp = np.random.randn(2000000, 3)
    data = lg.array(datanp)
    pointsnp = np.random.choice(lg.arange(len(data)), 4, False)
    points = lg.array(pointsnp)

    centroids = data[points]
    centroidsnp = datanp[pointsnp]
    sqdists = lg.zeros((4, len(data)))
    sqdistsnp = np.zeros((4, len(datanp)))
    for i in range(4):
        vec = data - centroids[i]
        vecnp = datanp - centroidsnp[i]
        sqdists[i] = lg.linalg.norm(vec, axis=1)
        sqdistsnp[i] = np.linalg.norm(vecnp, axis=1)

    clusters = lg.argmin(sqdists, axis=0)
    clustersnp = np.argmin(sqdistsnp, axis=0)
    assert lg.array_equal(lg.where(clusters == 0), np.where(clustersnp == 0))


if __name__ == "__main__":
    test()
