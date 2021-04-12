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

import legate.numpy as lg


def testtion():

    word_size = 10
    hidden_size = 10
    sentence_length = 5
    batch_size = 3

    np.random.seed(42)

    WLSTM_np = np.random.randn(
        word_size + hidden_size, 4 * hidden_size
    ) / np.sqrt(word_size + hidden_size)

    xphpb = WLSTM_np.shape[0]
    d = hidden_size
    n = sentence_length
    b = batch_size

    WLSTM_lg = lg.array(WLSTM_np)

    dHout_np = np.random.randn(n, b, d)
    IFOGf_np = np.random.randn(n, b, d * 4)
    C_np = np.random.randn(n, b, d)
    Ct_np = np.random.randn(n, b, d)
    Hin_np = np.random.randn(n, b, xphpb)

    dIFOG_np = np.zeros((n, b, d * 4))
    dIFOGf_np = np.zeros(IFOGf_np.shape)
    dHin_np = np.zeros(Hin_np.shape)
    dC_np = np.zeros(C_np.shape)
    dh0_np = np.zeros((1, d))

    dHout_lg = lg.array(dHout_np)
    IFOGf_lg = lg.array(IFOGf_np)
    C_lg = lg.array(C_np)
    Ct_lg = lg.array(Ct_np)
    Hin_lg = lg.array(Hin_np)

    dIFOG_lg = lg.zeros((n, b, d * 4))
    dIFOGf_lg = lg.zeros(IFOGf_lg.shape)
    dHin_lg = lg.zeros(Hin_lg.shape)
    dC_lg = lg.zeros(C_lg.shape)
    dh0_lg = lg.zeros((1, d))

    for t in reversed(range(n)):
        tanhCt_np = Ct_np[t]
        tanhCt_lg = Ct_lg[t]
        # assert lg.allclose(tanhCt_np, tanhCt_lg)

        dIFOGf_np[t, :, 2 * d : 3 * d] = tanhCt_np * dHout_np[t]
        dIFOGf_lg[t, :, 2 * d : 3 * d] = tanhCt_lg * dHout_lg[t]
        # assert lg.allclose(dIFOGf_np[t,:,2*d:3*d], dIFOGf_lg[t,:,2*d:3*d])

        # backprop tanh non-linearity first then continue backprop
        dC_np[t] += (1 - tanhCt_np ** 2) * (
            IFOGf_np[t, :, 2 * d : 3 * d] * dHout_np[t]
        )
        dC_lg[t] += (1 - tanhCt_lg ** 2) * (
            IFOGf_lg[t, :, 2 * d : 3 * d] * dHout_lg[t]
        )
        # assert lg.allclose(dC_np[t], dC_lg[t])

        if t > 0:
            dIFOGf_np[t, :, d : 2 * d] = C_np[t - 1] * dC_np[t]
            dIFOGf_lg[t, :, d : 2 * d] = C_lg[t - 1] * dC_lg[t]
            # assert lg.allclose(dIFOGf_np[t,:,d:2*d], dIFOGf_lg[t,:,d:2*d])

            dC_np[t - 1] += IFOGf_np[t, :, d : 2 * d] * dC_np[t]
            dC_lg[t - 1] += IFOGf_lg[t, :, d : 2 * d] * dC_lg[t]
            # assert lg.allclose(dC_np[t-1], dC_lg[t-1])

        dIFOGf_np[t, :, :d] = IFOGf_np[t, :, 3 * d :] * dC_np[t]
        dIFOGf_lg[t, :, :d] = IFOGf_lg[t, :, 3 * d :] * dC_lg[t]
        # assert lg.allclose(dIFOGf_np[t,:,:d], dIFOGf_lg[t,:,:d])

        dIFOGf_np[t, :, 3 * d :] = IFOGf_np[t, :, :d] * dC_np[t]
        dIFOGf_lg[t, :, 3 * d :] = IFOGf_lg[t, :, :d] * dC_lg[t]
        # assert lg.allclose(dIFOGf_np, dIFOGf_lg)

        # backprop activation functions
        dIFOG_np[t, :, 3 * d :] = (
            1 - IFOGf_np[t, :, 3 * d :] ** 2
        ) * dIFOGf_np[t, :, 3 * d :]
        dIFOG_lg[t, :, 3 * d :] = (
            1 - IFOGf_lg[t, :, 3 * d :] ** 2
        ) * dIFOGf_lg[t, :, 3 * d :]
        # assert lg.allclose(dIFOG_np[t,:,3*d:], dIFOG_lg[t,:,3*d:])

        y_np = IFOGf_np[t, :, : 3 * d]
        y_lg = IFOGf_lg[t, :, : 3 * d]
        # assert lg.allclose(y_np, y_lg)

        dIFOG_np[t, :, : 3 * d] = (y_np * (1.0 - y_np)) * dIFOGf_np[
            t, :, : 3 * d
        ]
        dIFOG_lg[t, :, : 3 * d] = (y_lg * (1.0 - y_lg)) * dIFOGf_lg[
            t, :, : 3 * d
        ]
        # assert lg.allclose(dIFOG_np[t,:,:3*d], dIFOG_lg[t,:,:3*d])

        # backprop matrix multiply
        dHin_np[t] = dIFOG_np[t].dot(WLSTM_np.transpose())
        dHin_lg[t] = dIFOG_lg[t].dot(WLSTM_lg.transpose())
        # assert lg.allclose(dHin_np[t], dHin_lg[t])

        # backprop the identity transforms into Hin
        if t > 0:
            dHout_np[t - 1, :] += dHin_np[t, :, word_size:]
            dHout_lg[t - 1, :] += dHin_lg[t, :, word_size:]
            # assert lg.allclose(dHout_np[t-1,:], dHout_lg[t-1,:])
        else:
            dh0_np[0] += np.sum(dHin_np[t, :, word_size:], 0)
            dh0_lg[0] += lg.sum(dHin_lg[t, :, word_size:], 0)
            # Check this one at the end
    # print(dh0_np[0])
    # print(dh0_lg[0])
    assert np.allclose(dh0_np[0], dh0_lg[0])


if __name__ == "__main__":
    testtion()
