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

from __future__ import print_function

import argparse
import datetime

from benchmark import run_benchmark

import cunumeric as np


# This is technically dead code right now, but we'll keep it around in
# case we want to generate a symmetrix positive definite matrix later
def generate_random(N):
    print("Generating %dx%d symmetric positive definite system..." % (N, N))
    # Generate a random matrix
    A = np.random.rand(N, N)
    # Make sure the matrix is symmetric
    A = 0.5 * (A + A.T)
    # Now make sure that it is positive definite
    A = A + N * np.eye(N)
    b = np.random.rand(N)
    return A, b


def generate_2D(N, corners):
    if corners:
        print(
            "Generating %dx%d 2-D adjacency system with corners..."
            % (N ** 2, N ** 2)
        )
        A = np.zeros((N ** 2, N ** 2)) + 8 * np.eye(N ** 2)
    else:
        print(
            "Generating %dx%d 2-D adjacency system without corners..."
            % (N ** 2, N ** 2)
        )
        A = np.zeros((N ** 2, N ** 2)) + 4 * np.eye(N ** 2)
    # These are the same for both cases
    off_one = np.full(N ** 2 - 1, -1, dtype=np.float64)
    A += np.diag(off_one, k=1)
    A += np.diag(off_one, k=-1)
    off_N = np.full(N * (N - 1), -1, dtype=np.float64)
    A += np.diag(off_N, k=N)
    A += np.diag(off_N, k=-N)
    # If we have corners then we have four more cases
    if corners:
        off_N_plus = np.full(N * (N - 1) - 1, -1, dtype=np.float64)
        A += np.diag(off_N_plus, k=N + 1)
        A += np.diag(off_N_plus, k=-(N + 1))
        off_N_minus = np.full(N * (N - 1) + 1, -1, dtype=np.float64)
        A += np.diag(off_N_minus, k=N - 1)
        A += np.diag(off_N_minus, k=-(N - 1))
    # Then we can generate a random b matrix
    b = np.random.rand(N ** 2)
    return A, b


def solve(A, b, conv_iters, max_iters, verbose):
    print("Solving system...")
    x = np.zeros(A.shape[1])
    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)
    converged = -1
    # Should always converge in fewer iterations than this
    max_iters = (
        min(max_iters, b.shape[0]) if max_iters is not None else b.shape[0]
    )
    for i in range(max_iters):
        Ap = A.dot(p)
        alpha = rsold / (p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        # We only do the convergence test every conv_iters or on the last
        # iteration
        if (i % conv_iters == 0 or i == (max_iters - 1)) and np.sqrt(
            rsnew
        ) < 1e-10:
            converged = i
            break
        if verbose:
            print("Residual: " + str(rsnew))
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    if converged < 0:
        print("Convergence FAILURE!")
    else:
        print("Converged in %d iterations" % (converged))
    return x


def precondition(A, N, corners):
    if corners:
        d = 8 * (N ** 2)
    else:
        d = 4 * (N ** 2)
    M = np.diag(np.full(N ** 2, 1.0 / d))
    return M


def preconditioned_solve(A, M, b, conv_iters, max_iters, verbose):
    print("Solving system with preconditioner...")
    x = np.zeros(A.shape[1])
    r = b - A.dot(x)
    z = M.dot(r)
    p = z
    rzold = r.dot(z)
    converged = -1
    # Should always converge in fewer iterations than this
    max_iters = (
        min(max_iters, b.shape[0]) if max_iters is not None else b.shape[0]
    )
    for i in range(max_iters):
        Ap = A.dot(p)
        alpha = rzold / (p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        rznew = r.dot(r)
        # We only do the convergence test every conv_iters or on the
        # last iteration
        if (i % conv_iters == 0 or i == (max_iters - 1)) and np.sqrt(
            rznew
        ) < 1e-10:
            converged = i
            break
        if verbose:
            print("Residual: " + str(rznew))
        z = M.dot(r)
        rznew = r.dot(z)
        beta = rznew / rzold
        p = z + beta * p
        rzold = rznew
    if converged < 0:
        print("Convergence FAILURE!")
    else:
        print("Converged in %d iterations" % (converged))
    return x


def check(A, x, b):
    print("Checking result...")
    if np.allclose(A.dot(x), b):
        print("PASS!")
    else:
        print("FAIL!")


def run_cg(
    N,
    corners,
    preconditioner,
    conv_iters,
    max_iters,
    perform_check,
    timing,
    verbose,
):
    start = datetime.datetime.now()
    # A, b = generate_random(N)
    A, b = generate_2D(N, corners)
    if preconditioner:
        M = precondition(A, N, corners)
        x = preconditioned_solve(A, M, b, conv_iters, max_iters, verbose)
    else:
        x = solve(A, b, conv_iters, max_iters, verbose)
    if perform_check:
        check(A, x, b)
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    if timing:
        print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--check",
        dest="check",
        action="store_true",
        help="check the result of the solve",
    )
    parser.add_argument(
        "-k",
        "--corners",
        dest="corners",
        action="store_true",
        help="include corners when generating the adjacency matrix",
    )
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=25,
        dest="conv_iters",
        help="iterations between convergence tests",
    )
    parser.add_argument(
        "-p",
        "--pre",
        dest="precondition",
        action="store_true",
        help="use a Jacobi preconditioner",
    )
    parser.add_argument(
        "-m",
        "--max",
        type=int,
        default=None,
        dest="max_iters",
        help="bound the maximum number of iterations",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="print verbose output",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 - "
        "normal execution)",
    )

    args = parser.parse_args()
    run_benchmark(
        run_cg,
        args.benchmark,
        "PreCG" if args.precondition else "CG",
        (
            args.N,
            args.corners,
            args.precondition,
            args.conv_iters,
            args.max_iters,
            args.check,
            args.timing,
            args.verbose,
        ),
    )
