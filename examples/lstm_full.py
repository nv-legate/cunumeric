#!/usr/bin/env python

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

import argparse

from benchmark import parse_args, run_benchmark


class Param:
    def __init__(self, name, value):
        self.name = name
        self.v = value  # parameter value
        self.d = np.zeros_like(value)  # derivative
        self.m = np.zeros_like(value)  # momentum for AdaGrad


class Parameters:
    def __init__(self, H_size, X_size, z_size, weight_sd):
        self.W_f = Param(
            "W_f", np.random.randn(H_size, z_size) * weight_sd + 0.5
        )
        self.b_f = Param("b_f", np.zeros((H_size, 1)))

        self.W_i = Param(
            "W_i", np.random.randn(H_size, z_size) * weight_sd + 0.5
        )
        self.b_i = Param("b_i", np.zeros((H_size, 1)))

        self.W_C = Param("W_C", np.random.randn(H_size, z_size) * weight_sd)
        self.b_C = Param("b_C", np.zeros((H_size, 1)))

        self.W_o = Param(
            "W_o", np.random.randn(H_size, z_size) * weight_sd + 0.5
        )
        self.b_o = Param("b_o", np.zeros((H_size, 1)))

        # For final layer to predict the next character
        self.W_v = Param("W_v", np.random.randn(X_size, H_size) * weight_sd)
        self.b_v = Param("b_v", np.zeros((X_size, 1)))

    def all(self):
        return [
            self.W_f,
            self.W_i,
            self.W_C,
            self.W_o,
            self.W_v,
            self.b_f,
            self.b_i,
            self.b_C,
            self.b_o,
            self.b_v,
        ]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    return 1 - y * y


def forward(x, h_prev, C_prev, H_size, X_size, p):
    assert x.shape == (X_size, 1)
    assert h_prev.shape == (H_size, 1)
    assert C_prev.shape == (H_size, 1)

    z = np.row_stack((h_prev, x))
    f = sigmoid(np.dot(p.W_f.v, z) + p.b_f.v)
    i = sigmoid(np.dot(p.W_i.v, z) + p.b_i.v)
    C_bar = tanh(np.dot(p.W_C.v, z) + p.b_C.v)

    C = f * C_prev + i * C_bar
    o = sigmoid(np.dot(p.W_o.v, z) + p.b_o.v)
    h = o * tanh(C)

    v = np.dot(p.W_v.v, h) + p.b_v.v
    y = np.exp(v) / np.sum(np.exp(v))  # softmax

    return z, f, i, C_bar, C, o, h, v, y


def backward(
    target,
    dh_next,
    dC_next,
    C_prev,
    H_size,
    X_size,
    z,
    f,
    i,
    C_bar,
    C,
    o,
    h,
    v,
    y,
    p,
):
    assert z.shape == (X_size + H_size, 1)
    assert v.shape == (X_size, 1)
    assert y.shape == (X_size, 1)

    for param in [dh_next, dC_next, C_prev, f, i, C_bar, C, o, h]:
        assert param.shape == (H_size, 1)

    dv = np.copy(y)
    dv[target] -= 1

    p.W_v.d += np.dot(dv, h.T)
    p.b_v.d += dv

    dh = np.dot(p.W_v.v.T, dv)
    dh += dh_next
    do = dh * tanh(C)
    do = dsigmoid(o) * do
    p.W_o.d += np.dot(do, z.T)
    p.b_o.d += do

    dC = np.copy(dC_next)
    dC += dh * o * dtanh(tanh(C))
    dC_bar = dC * i
    dC_bar = dtanh(C_bar) * dC_bar
    p.W_C.d += np.dot(dC_bar, z.T)
    p.b_C.d += dC_bar

    di = dC * C_bar
    di = dsigmoid(i) * di
    p.W_i.d += np.dot(di, z.T)
    p.b_i.d += di

    df = dC * C_prev
    df = dsigmoid(f) * df
    p.W_f.d += np.dot(df, z.T)
    p.b_f.d += df

    dz = (
        np.dot(p.W_f.v.T, df)
        + np.dot(p.W_i.v.T, di)
        + np.dot(p.W_C.v.T, dC_bar)
        + np.dot(p.W_o.v.T, do)
    )
    dh_prev = dz[:H_size, :]
    dC_prev = f * dC

    return dh_prev, dC_prev


def clear_gradients(params):
    for p in params.all():
        p.d.fill(0)


def clip_gradients(params):
    for p in params.all():
        np.clip(p.d, -1, 1, out=p.d)


def forward_backward(
    inputs, targets, h_prev, C_prev, T_steps, H_size, X_size, parameters
):
    # To store the values for each time step
    (
        x_s,
        z_s,
        f_s,
        i_s,
    ) = (
        {},
        {},
        {},
        {},
    )
    C_bar_s, C_s, o_s, h_s = {}, {}, {}, {}
    v_s, y_s = {}, {}

    # Values at t - 1
    h_s[-1] = np.copy(h_prev)
    C_s[-1] = np.copy(C_prev)

    loss = 0
    # Loop through time steps
    assert len(inputs) == T_steps
    for t in range(len(inputs)):
        x_s[t] = np.zeros((X_size, 1))
        x_s[t][inputs[t]] = 1  # Input character

        (
            z_s[t],
            f_s[t],
            i_s[t],
            C_bar_s[t],
            C_s[t],
            o_s[t],
            h_s[t],
            v_s[t],
            y_s[t],
        ) = forward(
            x_s[t], h_s[t - 1], C_s[t - 1], H_size, X_size, parameters
        )  # Forward pass

        loss += -np.log(y_s[t][targets[t], 0])  # Loss for at t

    clear_gradients(parameters)

    dh_next = np.zeros_like(h_s[0])  # dh from the next character
    dC_next = np.zeros_like(C_s[0])  # dh from the next character

    for t in reversed(range(len(inputs))):
        # Backward pass
        dh_next, dC_next = backward(
            target=targets[t],
            dh_next=dh_next,
            dC_next=dC_next,
            C_prev=C_s[t - 1],
            H_size=H_size,
            X_size=X_size,
            z=z_s[t],
            f=f_s[t],
            i=i_s[t],
            C_bar=C_bar_s[t],
            C=C_s[t],
            o=o_s[t],
            h=h_s[t],
            v=v_s[t],
            y=y_s[t],
            p=parameters,
        )

    clip_gradients(parameters)

    return loss, h_s[len(inputs) - 1], C_s[len(inputs) - 1]


def update_parameters(learning_rate, params):
    for p in params.all():
        p.m += p.d * p.d  # Calculate sum of gradients
        # print(learning_rate * dparam)
        p.v += -(learning_rate * p.d / np.sqrt(p.m + 1e-8))


def update_status(iteration, smooth_loss):
    print("iter %d, loss %f" % (iteration, smooth_loss))


def run_lstm(
    file_name,
    H_size,
    T_steps,
    max_iters,
    learning_rate,
    weight_sd,
    dump,
    timing,
):
    with open(file_name, "r") as f:
        data = f.read()
        chars = list(set(data))
        data_size, X_size = len(data), len(chars)
        print("data has %d characters, %d unique" % (data_size, X_size))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}

    z_size = H_size + X_size  # Size of concatenate(H, X) vector

    parameters = Parameters(H_size, X_size, z_size, weight_sd)

    # Exponential average of loss
    # Initialize to a error of a random model
    smooth_loss = -np.log(1.0 / X_size) * T_steps

    pointer = 0

    timer.start()

    for iteration in range(max_iters):
        # Reset
        if pointer + T_steps >= len(data) or iteration == 0:
            g_h_prev = np.zeros((H_size, 1))
            g_C_prev = np.zeros((H_size, 1))
            pointer = 0

        inputs = [char_to_idx[ch] for ch in data[pointer : pointer + T_steps]]
        targets = [
            char_to_idx[ch] for ch in data[pointer + 1 : pointer + T_steps + 1]
        ]

        loss, g_h_prev, g_C_prev = forward_backward(
            inputs,
            targets,
            g_h_prev,
            g_C_prev,
            T_steps,
            H_size,
            X_size,
            parameters,
        )
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # Print every hundred steps
        if iteration % dump == 0:
            update_status(iteration, smooth_loss)

        update_parameters(learning_rate, parameters)

        pointer += T_steps
    update_status(max_iters, smooth_loss)

    total = timer.stop()
    if timing:
        print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dump",
        type=int,
        default=100,
        dest="dump",
        help="how many iterations of training between dumping output",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="input.txt",
        dest="file_name",
        help="input file name",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=100,
        dest="hidden",
        help="size of hidden layer",
    )
    parser.add_argument(
        "-l",
        "--loops",
        type=int,
        default=10000,
        dest="loops",
        help="maximum number of training loops to run",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=float,
        default=1e-1,
        dest="rate",
        help="learning rate",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=25,
        dest="steps",
        help="number of time steps (length of the sequence) used for training",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=float,
        default=0.1,
        dest="weight",
        help="standard deviation of weights for initialization",
    )

    args, np, timer = parse_args(parser)

    run_benchmark(
        run_lstm,
        args.benchmark,
        "LSTM Full",
        (
            args.file_name,
            args.hidden,
            args.steps,
            args.loops,
            args.rate,
            args.weight,
            args.dump,
            args.timing,
        ),
    )
