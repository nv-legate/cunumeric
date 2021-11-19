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
import glob
import json
import multiprocessing
import os
import platform
import subprocess
import sys

# Find physical core count of the machine.
if platform.system() == "Linux":
    lines = subprocess.check_output(["lscpu", "--parse=core"])
    physical_cores = len(
        set(
            line
            for line in lines.decode("utf-8").strip().split("\n")
            if not line.startswith("#")
        )
    )
elif platform.system() == "Darwin":
    physical_cores = int(
        subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"])
    )
else:
    raise Exception("Unknown platform: %s" % platform.system())

# Choose a reasonable number of application cores given the
# available physical cores.
app_cores = max(physical_cores - 2, 1)

# draw tests from these directories
legate_tests = []
legate_tests.extend(glob.glob("tests/*.py"))
legate_tests.extend(glob.glob("examples/*.py"))

# some test programs have additional command line arguments
test_flags = {
    "examples/lstm_full.py": ["--file", "resources/lstm_input.txt"],
}

# some tests are currently disabled
disabled_tests = [
    "examples/kmeans_sort.py",
    "examples/wgrad.py",
    "tests/reduction_axis.py",
    "examples/lstm_full.py",
    "examples/ingest.py",
]

# filter out disabled tests
legate_tests = sorted(
    filter(lambda test: test not in disabled_tests, legate_tests)
)

red = "\033[1;31m"
green = "\033[1;32m"
clear = "\033[0m"

FNULL = open(os.devnull, "w")


def load_json_config(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except IOError:
        return None


def cmd(command, env=None, cwd=None, stdout=None, stderr=None, show=True):
    if show:
        print(*command)
        sys.stdout.flush()
    return subprocess.check_call(
        command, env=env, cwd=cwd, stdout=stdout, stderr=stderr
    )


def run_test(
    test_file,
    driver,
    flags,
    test_flags,
    opts,
    env,
    root_dir,
    verbose,
):
    test_path = os.path.join(root_dir, test_file)
    try:
        cmd(
            [driver, test_path, "-cunumeric:test"] + flags + test_flags + opts,
            env=env,
            cwd=root_dir,
            stdout=FNULL if not verbose else sys.stderr,
            stderr=FNULL if not verbose else sys.stderr,
            show=verbose,
        )
        return True, test_file
    except (OSError, subprocess.CalledProcessError):
        return False, test_file


def report_result(test_name, result):
    (passed, test_file) = result

    if passed:
        print("[%sPASS%s] (%s) %s" % (green, clear, test_name, test_file))
        return 1
    else:
        print("[%sFAIL%s] (%s) %s" % (red, clear, test_name, test_file))
        return 0


def compute_thread_pool_size_for_gpu_tests(pynvml, gpus_per_test):
    # TODO: We need to make this configurable
    MEMORY_BUDGET = 6 << 30
    gpu_count = pynvml.nvmlDeviceGetCount()
    parallelism_per_gpu = 16
    for idx in range(gpu_count):
        info = pynvml.nvmlDeviceGetMemoryInfo(
            pynvml.nvmlDeviceGetHandleByIndex(idx)
        )
        parallelism_per_gpu = min(
            parallelism_per_gpu, info.free // MEMORY_BUDGET
        )

    return (
        parallelism_per_gpu * (gpu_count // gpus_per_test),
        parallelism_per_gpu,
    )


def run_test_legate(
    test_name,
    root_dir,
    legate_dir,
    flags,
    env,
    verbose,
    opts,
    workers,
    num_procs,
):
    if test_name == "GPU":
        try:
            import pynvml

            pynvml.nvmlInit()
        except ModuleNotFoundError:
            pynvml = None

    if workers is None:
        if verbose:
            workers = 1
        elif test_name != "GPU":
            workers = multiprocessing.cpu_count() // num_procs
        else:
            if pynvml is None:
                workers = 1
            else:
                threads, parallelism = compute_thread_pool_size_for_gpu_tests(
                    pynvml, num_procs
                )
                gpu_count = pynvml.nvmlDeviceGetCount()
                workers = threads * num_procs // gpu_count

    driver = os.path.join(legate_dir, "bin", "legate")
    total_pass = 0
    if workers == 1:
        for test_file in legate_tests:
            result = run_test(
                test_file,
                driver,
                flags,
                test_flags.get(test_file, []),
                opts,
                env,
                root_dir,
                verbose,
            )
            total_pass += report_result(test_name, result)
    else:
        # Turn off the core pinning so that the tests can run concurrently
        env["REALM_SYNTHETIC_CORE_MAP"] = ""

        pool = multiprocessing.Pool(workers)

        results = []
        for idx, test_file in enumerate(legate_tests):
            results.append(
                pool.apply_async(
                    run_test,
                    (
                        test_file,
                        driver,
                        flags,
                        test_flags.get(test_file, []),
                        opts,
                        env,
                        root_dir,
                        verbose,
                    ),
                )
            )

        pool.close()

        result_set = set(results)
        while True:
            completed = set()
            for result in result_set:
                if result.ready():
                    total_pass += report_result(test_name, result.get())
                    completed.add(result)
            result_set -= completed
            if len(result_set) == 0:
                break

    print(
        "%24s: Passed %4d of %4d tests (%5.1f%%)"
        % (
            "%s" % test_name,
            total_pass,
            len(legate_tests),
            float(100 * total_pass) / len(legate_tests),
        )
    )
    return total_pass


def option_enabled(option, options, var_prefix="", default=True):
    if options is not None:
        return option in options
    option_var = "%s%s" % (var_prefix, option.upper())
    if option_var in os.environ:
        return os.environ[option_var] == "1"
    return default


class Stage(object):
    __slots__ = ["name", "begin_time"]

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin_time = datetime.datetime.now()
        print()
        print("#" * 60)
        print("### Entering Stage: %s" % self.name)
        print("#" * 60)
        print()
        sys.stdout.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.datetime.now()
        print()
        print("#" * 60)
        print("### Exiting Stage: %s" % self.name)
        print("###   * Exception Type: %s" % exc_type)
        print("###   * Elapsed Time: %s" % (end_time - self.begin_time))
        print("#" * 60)
        print()
        sys.stdout.flush()


def report_mode(
    debug,
    use_gasnet,
    use_cuda,
    use_openmp,
    use_llvm,
    use_hdf,
    use_spy,
    use_gcov,
    use_cmake,
    use_cpus,
    interop_tests,
):
    print()
    print("#" * 60)
    print("### Test Suite Configuration")
    print("###")
    print("### Debug:               %s" % debug)
    print("###")
    print("### Test Flags:")
    print("###   * CPUs:            %s" % use_cpus)
    print("###   * GASNet:          %s" % use_gasnet)
    print("###   * CUDA:            %s" % use_cuda)
    print("###   * OpenMP:          %s" % use_openmp)
    print("###   * LLVM:            %s" % use_llvm)
    print("###   * HDF5:            %s" % use_hdf)
    print("###   * Spy:             %s" % use_spy)
    print("###   * Gcov:            %s" % use_gcov)
    print("###   * CMake:           %s" % use_cmake)
    print("###")
    print("### Integration Tests:   %s" % interop_tests)
    print("###")
    print("#" * 60)
    print()
    sys.stdout.flush()


def run_tests(
    debug=True,
    use_features=None,
    cpus=None,
    gpus=None,
    openmp=None,
    ompthreads=None,
    root_dir=None,
    legate_dir=None,
    verbose=False,
    options=[],
    interop_tests=False,
    workers=None,
):

    if interop_tests:
        legate_tests.extend(glob.glob("tests/interop/*.py"))

    if root_dir is None:
        root_dir = os.path.dirname(os.path.realpath(__file__))

    if legate_dir is None:
        legate_config = os.path.join(root_dir, ".legate.core.json")
        if "LEGATE_DIR" in os.environ:
            legate_dir = os.environ["LEGATE_DIR"]
        elif legate_dir is None:
            legate_dir = load_json_config(legate_config)
        if legate_dir is None or not os.path.exists(legate_dir):
            raise Exception(
                "You need to provide a Legate installation "
                + "directory using '--legate'"
            )
        legate_dir = os.path.realpath(legate_dir)

    # Determine which features to test with.
    def feature_enabled(feature, default=True):
        return option_enabled(feature, use_features, "USE_", default)

    use_gasnet = feature_enabled("gasnet", False)
    use_cuda = feature_enabled("cuda", False)
    use_openmp = feature_enabled("openmp", False)
    use_llvm = feature_enabled("llvm", False)
    use_hdf = feature_enabled("hdf", False)
    use_spy = feature_enabled("spy", False)
    use_gcov = feature_enabled("gcov", False)
    use_cmake = feature_enabled("cmake", False)
    use_cpus = feature_enabled("cpus", False)

    if not (use_cpus or use_cuda or use_openmp):
        use_cpus = True

    gcov_flags = " -ftest-coverage -fprofile-arcs"

    report_mode(
        debug,
        use_gasnet,
        use_cuda,
        use_openmp,
        use_llvm,
        use_hdf,
        use_spy,
        use_gcov,
        use_cmake,
        use_cpus,
        interop_tests,
    )

    # Normalize the test environment.
    env = dict(
        list(os.environ.items())
        + [
            ("LEGATE_TEST", "1"),
            ("DEBUG", "1" if debug else "0"),
            ("USE_GASNET", "1" if use_gasnet else "0"),
            ("USE_CUDA", "1" if use_cuda else "0"),
            ("USE_OPENMP", "1" if use_openmp else "0"),
            ("USE_PYTHON", "1"),  # Always need python for Legate
            ("USE_LLVM", "1" if use_llvm else "0"),
            ("USE_HDF", "1" if use_hdf else "0"),
            ("USE_SPY", "1" if use_spy else "0"),
        ]
        + (
            # Gcov doesn't get a USE_GCOV flag, but instead stuff the GCC
            # options for Gcov on to the compile and link flags.
            [
                (
                    "CC_FLAGS",
                    (
                        os.environ["CC_FLAGS"] + gcov_flags
                        if "CC_FLAGS" in os.environ
                        else gcov_flags
                    ),
                ),
                (
                    "LD_FLAGS",
                    (
                        os.environ["LD_FLAGS"] + gcov_flags
                        if "LD_FLAGS" in os.environ
                        else gcov_flags
                    ),
                ),
            ]
            if use_gcov
            else []
        )
    )

    total_pass, total_count = 0, 0
    if use_cpus:
        with Stage("CPU tests"):
            count = run_test_legate(
                "CPU",
                root_dir,
                legate_dir,
                ["--cpus", str(cpus)],
                env,
                verbose,
                options,
                workers,
                cpus,
            )
            total_pass += count
            total_count += len(legate_tests)
    if use_cuda:
        with Stage("GPU tests"):
            count = run_test_legate(
                "GPU",
                root_dir,
                legate_dir,
                ["--gpus", str(gpus)],
                env,
                verbose,
                options,
                workers,
                gpus,
            )
            total_pass += count
            total_count += len(legate_tests)
    if use_openmp:
        with Stage("OpenMP tests"):
            count = run_test_legate(
                "OMP",
                root_dir,
                legate_dir,
                ["--omps", str(openmp), "--ompthreads", str(ompthreads)],
                env,
                verbose,
                options,
                workers,
                openmp * ompthreads,
            )
            total_pass += count
            total_count += len(legate_tests)
    print("    " + "~" * 54)
    print(
        "%24s: Passed %4d of %4d tests (%5.1f%%)"
        % (
            "total",
            total_pass,
            total_count,
            (float(100 * total_pass) / total_count),
        )
    )
    return not (total_count == total_pass)


# behaves enough like a normal list for ArgumentParser's needs, except for
#  the __contains__ method, which accepts a list of values and checks each
#  one for membership
class MultipleChoiceList(object):
    def __init__(self, *args):
        self.list = list(args)

    def __contains__(self, x):
        if type(x) is list:
            for v in x:
                if v not in self.list:
                    return False
            return True
        else:
            return x in self.list

    def __iter__(self):
        return self.list.__iter__()


class ExtendAction(argparse.Action):
    def __init__(self, **kwargs):
        super(ExtendAction, self).__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = items[:] if items else []
        if type(values) is list:
            items.extend(values)
        else:
            items.append(values)
        setattr(namespace, self.dest, items)


def driver():
    parser = argparse.ArgumentParser(
        description="Legate test suite",
        epilog="Any unrecognized arguments will be forwarded to the Legate "
        "driver script",
    )

    # Run options:
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=os.environ["DEBUG"] == "1" if "DEBUG" in os.environ else True,
        help="Invoke Legate in debug mode (also via DEBUG).",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable debug mode (equivalent to DEBUG=0).",
    )
    parser.add_argument(
        "--use",
        dest="use_features",
        action=ExtendAction,
        choices=MultipleChoiceList(
            "gasnet",
            "cuda",
            "openmp",
            "llvm",
            "hdf",
            "spy",
            "gcov",
            "cmake",
            "cpus",
        ),
        type=lambda s: s.split(","),
        help="Test Legate with features (also via USE_*).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        dest="cpus",
        help="Number of CPUs per node to use",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        dest="gpus",
        help="Number of GPUs per node to use",
    )
    parser.add_argument(
        "--omps",
        type=int,
        default=1,
        dest="openmp",
        help="Number OpenMP processors per node to use",
    )
    parser.add_argument(
        "--ompthreads",
        type=int,
        default=4,
        dest="ompthreads",
        help="Number of threads per OpenMP processor",
    )
    parser.add_argument(
        "-C",
        "--directory",
        dest="root_dir",
        metavar="DIR",
        action="store",
        required=False,
        help="Root directory from which to run tests.",
    )
    parser.add_argument(
        "--legate",
        dest="legate_dir",
        metavar="DIR",
        action="store",
        required=False,
        help="Legate installation directory.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Print more debugging information.",
    )
    parser.add_argument(
        "--interop",
        dest="interop_tests",
        action="store_true",
        help="Include integration tests with other Legate libraries.",
    )
    parser.add_argument(
        "-j",
        type=int,
        default=None,
        dest="workers",
        help="Number of parallel workers for testing",
    )

    args, opts = parser.parse_known_args()

    sys.exit(run_tests(options=opts, **vars(args)))


if __name__ == "__main__":
    driver()
