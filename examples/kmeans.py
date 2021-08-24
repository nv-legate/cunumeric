#!/usr/bin/env python

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

# Derived from https://github.com/bryancatanzaro/kmeans

from __future__ import print_function

import argparse
import datetime

from benchmark import run_benchmark

import legate.numpy as np


def initialize(N, D, C, T):
    # Uncomment this if we want execution to be deterministic with
    # original numpy np.random.seed(0)
    data = np.random.random((N, D)).astype(T)
    # Since points are random, we'll just generate some random centers
    centroids = np.random.random((C, D)).astype(T)
    return data, centroids


def calculate_distances(data, centroids, data_dots):
    centroid_dots = np.square(np.linalg.norm(centroids, ord=2, axis=1))
    pairwise_distances = (
        data_dots[:, np.newaxis] + centroid_dots[np.newaxis, :]
    )
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x . y
    # pairwise_distances has ||x||^2 + ||y||^2, so beta = 1
    # The gemm calculates x.y for all x and y, so alpha = -2.0
    pairwise_distances -= 2.0 * np.dot(data, centroids.T)
    return pairwise_distances


def relabel(pairwise_distances):
    return np.argmin(pairwise_distances, axis=1)


def find_centroids(centroids, data, labels, pairwise_distances, zero_point, C):
    # Get the number of points associated with each centroid
    counts = np.bincount(labels, minlength=C)

    # more bincounts using the positions as weights produce the unnormalized
    #  updated centroid locations (have to do each dimension separately since
    #  a weight cannot be a vector)
    for idx in range(data.shape[1]):
        centroids[:, idx] = np.bincount(
            labels, weights=data[:, idx], minlength=C
        )

    # would have been nice if numpy offered a combined amin/argmin to avoid
    #  iterating over pairwise_distances twice
    distance_sum = np.sum(np.amin(pairwise_distances, axis=1))

    # To avoid introducing divide by zero errors
    # If a centroid has no weight, we'll do no normalization
    # This will keep its coordinates defined.
    counts = np.maximum(counts, np.ones((1,), dtype=np.uint64))
    centroids /= counts[:, np.newaxis]
    return distance_sum


def run_kmeans(C, D, T, I, N, S, benchmarking):  # noqa: E741
    print("Running kmeans...")
    print("Number of data points: " + str(N))
    print("Number of dimensions: " + str(D))
    print("Number of centroids: " + str(C))
    print("Max iterations: " + str(I))
    start = datetime.datetime.now()
    data, centroids = initialize(N, D, C, T)

    data_dots = np.square(np.linalg.norm(data, ord=2, axis=1))
    zero_point = np.zeros((1, data.shape[1]), dtype=data.dtype)

    labels = None
    iteration = 0
    prior_distance_sum = None
    # We run for max iterations or until we converge
    # We only test convergence every S iterations
    while iteration < I:
        pairwise_distances = calculate_distances(data, centroids, data_dots)

        new_labels = relabel(pairwise_distances)

        distance_sum = find_centroids(
            centroids, data, new_labels, pairwise_distances, zero_point, C
        )

        if iteration > 0 and iteration % S == 0:
            changes = np.not_equal(labels, new_labels)
            total_changes = np.sum(changes)
            delta = distance_sum / prior_distance_sum
            print(
                "Iteration "
                + str(iteration)
                + " produced "
                + str(total_changes)
                + " changes, and total distance is "
                + str(distance_sum)
            )
            # We ignore the result of the threshold test in the case
            # that we are running performance benchmarks to measure
            # performance for a certain number of iterations
            if delta > 1 - 0.000001 and not benchmarking:
                print("Threshold triggered, terminating iterations early")
                break
        prior_distance_sum = distance_sum
        labels = new_labels
        iteration += 1
    # This final distance sum also synchronizes the results
    print(
        "Final distance sum at iteration "
        + str(iteration)
        + ": "
        + str(prior_distance_sum)
    )
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--centers",
        type=int,
        default=10,
        dest="C",
        help="number of centroids",
    )
    parser.add_argument(
        "-d",
        "--dims",
        type=int,
        default=2,
        dest="D",
        help="number of dimensions for each input data point",
    )
    parser.add_argument(
        "-m",
        "--max-iters",
        type=int,
        default=1000,
        dest="I",
        help="maximum number of iterations to run the algorithm for",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        dest="N",
        help="number of elements in the data set in thousands",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=32,
        dest="P",
        help="precision of the computation in bits",
    )
    parser.add_argument(
        "-s",
        "--sample",
        type=int,
        default=25,
        dest="S",
        help="number of iterations between sampling the log likelihood",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application"
        " (default 1 - normal execution)",
    )
    args = parser.parse_args()
    if args.P == 16:
        run_benchmark(
            run_kmeans,
            args.benchmark,
            "KMEANS(H)",
            (
                args.C,
                args.D,
                np.float16,
                args.I,
                args.N * 1000,
                args.S,
                args.benchmark > 1,
            ),
        )
    elif args.P == 32:
        run_benchmark(
            run_kmeans,
            args.benchmark,
            "KMEANS(S)",
            (
                args.C,
                args.D,
                np.float32,
                args.I,
                args.N * 1000,
                args.S,
                args.benchmark > 1,
            ),
        )
    elif args.P == 64:
        run_benchmark(
            run_kmeans,
            args.benchmark,
            "KMEANS(D)",
            (
                args.C,
                args.D,
                np.float64,
                args.I,
                args.N * 1000,
                args.S,
                args.benchmark > 1,
            ),
        )
    else:
        raise TypeError("Precision must be one of 16, 32, or 64")
