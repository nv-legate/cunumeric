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
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tempfile

_version = sys.version_info.major


if _version == 2:  # Python 2.x:
    _input = raw_input  # noqa: F821
elif _version == 3:  # Python 3.x:
    _input = input
else:
    raise Exception("Incompatible Python version")


os_name = platform.system()


if os_name == "Linux":
    dylib_ext = ".so"
elif os_name == "Darwin":
    dylib_ext = ".dylib"
else:
    raise Exception(
        "install.py script does not work on %s" % platform.system()
    )


def print_log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def execute_command(args, verbose, cwd=None, shell=False):
    if verbose:
        print_log("EXECUTING: ", args)
    subprocess.check_call(args, cwd=cwd, shell=shell)


def git_clone(repo_dir, url, verbose, branch=None, tag=None):
    assert branch is not None or tag is not None
    if branch is not None:
        execute_command(
            ["git", "clone", "-b", branch, url, repo_dir], verbose=verbose
        )
    else:
        execute_command(
            ["git", "clone", "--single-branch", "-b", tag, url, repo_dir],
            verbose=verbose,
        )
        execute_command(
            ["git", "checkout", "-b", "master"], cwd=repo_dir, verbose=verbose
        )


def git_reset(repo_dir, refspec, verbose):
    execute_command(
        ["git", "reset", "--hard", refspec], cwd=repo_dir, verbose=verbose
    )


def git_update(repo_dir, verbose, branch=None):
    execute_command(
        ["git", "pull", "--ff-only"], cwd=repo_dir, verbose=verbose
    )
    if branch is not None:
        execute_command(
            ["git", "checkout", branch], cwd=repo_dir, verbose=verbose
        )


def load_json_config(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except IOError:
        return None


def dump_json_config(filename, value):
    with open(filename, "w") as f:
        return json.dump(value, f)


def symlink(from_path, to_path):
    if not os.path.lexists(to_path):
        os.symlink(from_path, to_path)


def has_openmp():
    temp_dir = tempfile.mkdtemp()
    try:
        execute_command(
            'echo "int main(void) { return 0; }" | '
            "g++ -o test.omp -x c++ -fopenmp -",
            shell=True,
            cwd=temp_dir,
            verbose=False,
        )
    except subprocess.CalledProcessError:
        return False
    else:
        return True


def install_openblas(openblas_dir, thread_count, verbose):
    print_log("Legate is installing OpenBLAS into a local directory...")
    temp_dir = tempfile.mkdtemp()
    # Pin OpenBLAS at 3.10 for now
    git_clone(
        temp_dir,
        url="https://github.com/xianyi/OpenBLAS.git",
        tag="v0.3.13",
        verbose=verbose,
    )
    # We can just build this directly
    if has_openmp():
        execute_command(
            [
                "make",
                "-j",
                str(thread_count),
                "USE_THREAD=1",
                "NO_STATIC=1",
                "USE_OPENMP=1",
                "NUM_PARALLEL=32",
                "LIBNAMESUFFIX=legate",
            ],
            cwd=temp_dir,
            verbose=verbose,
        )
    else:
        execute_command(
            [
                "make",
                "-j",
                str(thread_count),
                "USE_THREAD=1",
                "NO_STATIC=1",
                "NUM_PARALLEL=32",
                "LIBNAMESUFFIX=legate",
            ],
            cwd=temp_dir,
            verbose=verbose,
        )
    # Then do the installation to our target directory
    execute_command(
        [
            "make",
            "-j",
            str(thread_count),
            "install",
            "PREFIX=" + openblas_dir,
            "LIBNAMESUFFIX=legate",
        ],
        cwd=temp_dir,
        verbose=verbose,
    )
    shutil.rmtree(temp_dir)


def get_cmake_config(cmake, legate_dir, default=None):
    config_filename = os.path.join(legate_dir, ".cmake.json")
    if cmake is None:
        cmake = load_json_config(config_filename)
        if cmake is None:
            cmake = default
    assert cmake in [True, False]
    dump_json_config(config_filename, cmake)
    return cmake


def find_c_define(define, header):
    with open(header, "r") as f:
        line = f.readline()
        while line:
            line = line.rstrip()
            if line.startswith("#define") and define in line.split(" "):
                return True
            line = f.readline()
    return False


def build_legate_numpy(
    legate_numpy_dir,
    install_dir,
    openblas_dir,
    cmake,
    cmake_exe,
    cuda,
    openmp,
    debug,
    debug_release,
    check_bounds,
    clean_first,
    python_only,
    thread_count,
    verbose,
    unknown,
):
    src_dir = os.path.join(legate_numpy_dir, "src")
    if cmake:
        print_log(
            "Warning: CMake is currently not supported for Legate NumPy build."
        )
        print_log("Using GNU Make for now.")

    if not python_only:
        openblas_dir = os.path.realpath(openblas_dir)
        install_dir = os.path.realpath(install_dir)
        if install_dir == os.path.commonprefix([openblas_dir, install_dir]):
            libname = "openblas_legate"
        else:
            libname = "openblas"

        make_flags = (
            [
                "LEGATE_DIR=%s" % install_dir,
                "OPEN_BLAS_DIR=%s" % openblas_dir,
                "DEBUG=%s" % (1 if debug else 0),
                "DEBUG_RELEASE=%s" % (1 if debug_release else 0),
                "CHECK_BOUNDS=%s" % (1 if check_bounds else 0),
                "PREFIX=%s" % install_dir,
                "OPENBLAS_FLAGS = -L%s/lib -l%s -Wl,-rpath=%s/lib"
                % (openblas_dir, libname, openblas_dir),
            ]
            + (["GCC=%s" % os.environ["CXX"]] if "CXX" in os.environ else [])
            + (["USE_CUDA=0"] if not cuda else [])
            + (["USE_OPENMP=0"] if not openmp else [])
        )
        if clean_first:
            execute_command(
                ["make"] + make_flags + ["clean"], cwd=src_dir, verbose=verbose
            )
        execute_command(
            ["make"] + make_flags + ["-j", str(thread_count), "install"],
            cwd=src_dir,
            verbose=verbose,
        )

    try:
        shutil.rmtree(os.path.join(legate_numpy_dir, "build"))
    except FileNotFoundError:
        pass

    cmd = ["python", "setup.py", "install", "--recurse"]
    if unknown is not None:
        cmd += unknown
        if "--prefix" not in unknown:
            cmd += ["--prefix", str(install_dir)]
    else:
        cmd += ["--prefix", str(install_dir)]
    execute_command(cmd, cwd=legate_numpy_dir, verbose=verbose)


def install_legate_numpy(
    cmake=None,
    cmake_exe=None,
    legate_dir=None,
    legion_dir=None,
    openblas_dir=None,
    thrust_dir=None,
    cuda=True,
    openmp=True,
    debug=False,
    debug_release=False,
    check_bounds=False,
    clean_first=False,
    python_only=False,
    thread_count=None,
    verbose=False,
    unknown=None,
):
    print_log("Verbose build is ", "on" if verbose else "off")
    if verbose:
        print_log("Options are:\n")
        print_log("cmake: ", cmake, "\n")
        print_log("cmake_exe: ", cmake_exe, "\n")
        print_log("legate_dir: ", legate_dir, "\n")
        print_log("legion_dir: ", legion_dir, "\n")
        print_log("openblas_dir: ", openblas_dir, "\n")
        print_log("cuda: ", cuda, "\n")
        print_log("openmp: ", openmp, "\n")
        print_log("debug: ", debug, "\n")
        print_log("debug_release: ", debug_release, "\n")
        print_log("check_bounds: ", check_bounds, "\n")
        print_log("clean_first: ", clean_first, "\n")
        print_log("python_only: ", python_only, "\n")
        print_log("thread_count: ", thread_count, "\n")
        print_log("verbose: ", verbose, "\n")
        print_log("unknown: ", unknown, "\n")

    legate_numpy_dir = os.path.dirname(os.path.realpath(__file__))

    cmake = get_cmake_config(cmake, legate_numpy_dir, default=False)

    if clean_first is None:
        clean_first = not cmake

    thread_count = thread_count
    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    # Check to see if we installed Legate Core
    legate_config = os.path.join(legate_numpy_dir, ".legate.core.json")
    if "LEGATE_DIR" in os.environ:
        legate_dir = os.environ["LEGATE_DIR"]
    elif legate_dir is None:
        legate_dir = load_json_config(legate_config)
    if legate_dir is None or not os.path.exists(legate_dir):
        raise Exception(
            "You need to provide a Legate Core installation using"
            " the '--with-core' flag"
        )
    legate_dir = os.path.realpath(legate_dir)
    dump_json_config(legate_config, legate_dir)

    # Check to see if we have an installation of openblas
    try:
        f = open(os.path.join(legate_dir, ".legate-libs.json"), "r")
        libs_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        libs_config = {}
    if "OPEN_BLAS_DIR" in os.environ:
        openblas_dir = os.environ["OPEN_BLAS_DIR"]
    elif openblas_dir is None:
        openblas_dir = libs_config.get("openblas")
        if openblas_dir is None:
            openblas_dir = os.path.join(legate_dir, "OpenBLAS")
    if not os.path.exists(openblas_dir):
        install_openblas(openblas_dir, thread_count, verbose)
    libs_config["openblas"] = openblas_dir
    with open(
        os.path.join(legate_dir, "share", "legate", ".legate-libs.json"), "w"
    ) as f:
        json.dump(libs_config, f)

    if not thrust_dir:
        try:
            f = open(
                os.path.join(legate_dir, "share", "legate", ".thrust.json"),
                "r",
            )
            thrust_dir = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            thrust_config = os.path.join(legate_numpy_dir, ".thrust.json")
            if "THRUST_PATH" in os.environ:
                thrust_dir = os.environ["THRUST_PATH"]
            elif thrust_dir is None:
                thrust_dir = load_json_config(thrust_config)
                if thrust_dir is None:
                    raise Exception(
                        "Could not find Thrust installation, "
                        "use '--with-thrust' to specify a location."
                    )
            thrust_dir = os.path.realpath(thrust_dir)
            dump_json_config(thrust_config, thrust_dir)
    os.environ["CC_FLAGS"] = (
        "-I" + thrust_dir + " " + os.environ.get("CC_FLAGS", "")
    )
    os.environ["NVCC_FLAGS"] = (
        "-I" + thrust_dir + " " + os.environ.get("NVCC_FLAGS", "")
    )

    build_legate_numpy(
        legate_numpy_dir,
        legate_dir,
        openblas_dir,
        cmake,
        cmake_exe,
        cuda,
        openmp,
        debug,
        debug_release,
        check_bounds,
        clean_first,
        python_only,
        thread_count,
        verbose,
        unknown,
    )


def driver():
    parser = argparse.ArgumentParser(description="Install Legate NumPy.")
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG") == "1",
        help="Build Legate NumPy with debugging enabled.",
    )
    parser.add_argument(
        "--debug-release",
        dest="debug_release",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG_RELEASE") == "1",
        help="Build Legate NumPy with debugging symbols.",
    )
    parser.add_argument(
        "--check-bounds",
        dest="check_bounds",
        action="store_true",
        required=False,
        default=False,
        help="Build Legate NumPy with bounds checks.",
    )
    parser.add_argument(
        "--with-core",
        dest="legate_dir",
        metavar="DIR",
        required=False,
        help="Path to Legate Core installation directory.",
    )
    parser.add_argument(
        "--with-openblas",
        dest="openblas_dir",
        metavar="DIR",
        required=False,
        help="Path to OpenBLAS installation directory. Note that providing a "
        "user-defined BLAS library may lead to dynamic library conflicts with "
        "BLAS loaded by Python's Numpy. When using legate.numpy's BLAS, this "
        "issue is prevented by a custom library name.",
    )
    parser.add_argument(
        "--with-thrust",
        dest="thrust_dir",
        metavar="DIR",
        required=False,
        help="Path to Thrust installation directory.",
    )
    parser.add_argument(
        "--no-cuda",
        dest="cuda",
        action="store_false",
        required=False,
        default=True,
        help="Build Legate NumPy without CUDA.",
    )
    parser.add_argument(
        "--no-openmp",
        dest="openmp",
        action="store_false",
        required=False,
        default=True,
        help="Build Legate NumPy without OpenMP.",
    )
    parser.add_argument(
        "--cmake",
        dest="cmake",
        action="store_true",
        required=False,
        default=os.environ["USE_CMAKE"] == "1"
        if "USE_CMAKE" in os.environ
        else None,
        help="Build Legate NumPy with CMake.",
    )
    parser.add_argument(
        "--no-cmake",
        dest="cmake",
        action="store_false",
        required=False,
        help="Don't build Legate NumPy with CMake (instead use GNU Make).",
    )
    parser.add_argument(
        "--with-cmake",
        dest="cmake_exe",
        metavar="EXE",
        required=False,
        default="cmake",
        help="Path to CMake executable (if not on PATH).",
    )
    parser.add_argument(
        "--no-clean",
        "--noclean",
        dest="clean_first",
        action="store_false",
        required=False,
        default=True,
        help="Skip clean before build.",
    )
    parser.add_argument(
        "--python-only",
        dest="python_only",
        action="store_true",
        required=False,
        default=False,
        help="Reinstall only the Python package.",
    )
    parser.add_argument(
        "-j",
        dest="thread_count",
        nargs="?",
        type=int,
        required=False,
        help="Number of threads used to compile.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose build output.",
    )
    args, unknown = parser.parse_known_args()

    install_legate_numpy(unknown=unknown, **vars(args))


if __name__ == "__main__":
    driver()
