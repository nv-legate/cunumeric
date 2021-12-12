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

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tempfile

# Flush output on newlines
sys.stdout.reconfigure(line_buffering=True)

os_name = platform.system()

if os_name == "Linux":
    dylib_ext = ".so"
elif os_name == "Darwin":
    dylib_ext = ".dylib"
else:
    raise Exception("install.py script does not work on %s" % os_name)


class BooleanFlag(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        default,
        required=False,
        help="",
        metavar=None,
    ):
        assert all(not opt.startswith("--no") for opt in option_strings)

        def flatten(list):
            return [item for sublist in list for item in sublist]

        option_strings = flatten(
            [
                [opt, "--no-" + opt[2:], "--no" + opt[2:]]
                if opt.startswith("--")
                else [opt]
                for opt in option_strings
            ]
        )
        super().__init__(
            option_strings,
            dest,
            nargs=0,
            const=None,
            default=default,
            type=bool,
            choices=None,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string):
        setattr(namespace, self.dest, not option_string.startswith("--no"))


def execute_command(args, verbose, cwd=None, shell=False):
    if verbose:
        print("EXECUTING: ", args)
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
    cxx = os.getenv("CXX", "g++")
    temp_dir = tempfile.mkdtemp()
    try:
        execute_command(
            'echo "int main(void) { return 0; }" | '
            f"{cxx} -o test.omp -x c++ -fopenmp -",
            shell=True,
            cwd=temp_dir,
            verbose=False,
        )
    except subprocess.CalledProcessError:
        return False
    else:
        return True


def install_openblas(openblas_dir, thread_count, verbose):
    print("Legate is installing OpenBLAS into a local directory...")
    temp_dir = tempfile.mkdtemp()
    # Pin OpenBLAS at a recent version
    git_clone(
        temp_dir,
        url="https://github.com/xianyi/OpenBLAS.git",
        tag="v0.3.15",
        verbose=verbose,
    )
    # We can just build this directly
    execute_command(
        [
            "make",
            "-j",
            str(thread_count),
            "USE_THREAD=1",
            "NO_STATIC=1",
            "USE_CUDA=0",
            "USE_OPENMP=%s" % (1 if has_openmp() else 0),
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


def find_c_define(define, header):
    with open(header, "r") as f:
        line = f.readline()
        while line:
            line = line.rstrip()
            if line.startswith("#define") and define in line.split(" "):
                return True
            line = f.readline()
    return False


def build_cunumeric(
    cunumeric_dir,
    install_dir,
    openblas_dir,
    cmake,
    cmake_exe,
    debug,
    debug_release,
    check_bounds,
    clean_first,
    python_only,
    thread_count,
    verbose,
    unknown,
):
    src_dir = os.path.join(cunumeric_dir, "src")
    if cmake:
        print("Warning: CMake is currently not supported for cuNumeric build.")
        print("Using GNU Make for now.")

    if not python_only:
        openblas_dir = os.path.realpath(openblas_dir)
        install_dir = os.path.realpath(install_dir)
        if install_dir == os.path.commonprefix([openblas_dir, install_dir]):
            libname = "openblas_legate"
        else:
            libname = "openblas"
        make_flags = [
            "LEGATE_DIR=%s" % install_dir,
            "OPEN_BLAS_DIR=%s" % openblas_dir,
            "DEBUG=%s" % (1 if debug else 0),
            "DEBUG_RELEASE=%s" % (1 if debug_release else 0),
            "CHECK_BOUNDS=%s" % (1 if check_bounds else 0),
            "PREFIX=%s" % install_dir,
            "OPENBLAS_FLAGS = -L%s/lib -l%s -Wl,-rpath,%s/lib"
            % (openblas_dir, libname, openblas_dir),
        ]
        if clean_first:
            execute_command(
                ["make"] + make_flags + ["clean"],
                cwd=src_dir,
                verbose=verbose,
            )
        execute_command(
            ["make"] + make_flags + ["-j", str(thread_count), "install"],
            cwd=src_dir,
            verbose=verbose,
        )

    try:
        shutil.rmtree(os.path.join(cunumeric_dir, "build"))
    except FileNotFoundError:
        pass

    cmd = [sys.executable, "setup.py", "install", "--recurse"]
    if unknown is not None:
        cmd += unknown
        if "--prefix" not in unknown:
            cmd += ["--prefix", str(install_dir)]
    else:
        cmd += ["--prefix", str(install_dir)]
    execute_command(cmd, cwd=cunumeric_dir, verbose=verbose)


def install_cunumeric(
    cmake,
    cmake_exe,
    legate_dir,
    openblas_dir,
    thrust_dir,
    debug,
    debug_release,
    check_bounds,
    clean_first,
    python_only,
    thread_count,
    verbose,
    unknown,
):
    print("Verbose build is ", "on" if verbose else "off")
    if verbose:
        print("Options are:\n")
        print("cmake: ", cmake, "\n")
        print("cmake_exe: ", cmake_exe, "\n")
        print("legate_dir: ", legate_dir, "\n")
        print("openblas_dir: ", openblas_dir, "\n")
        print("debug: ", debug, "\n")
        print("debug_release: ", debug_release, "\n")
        print("check_bounds: ", check_bounds, "\n")
        print("clean_first: ", clean_first, "\n")
        print("python_only: ", python_only, "\n")
        print("thread_count: ", thread_count, "\n")
        print("verbose: ", verbose, "\n")
        print("unknown: ", unknown, "\n")

    cunumeric_dir = os.path.dirname(os.path.realpath(__file__))

    cmake_config = os.path.join(cunumeric_dir, ".cmake.json")
    dump_json_config(cmake_config, cmake)

    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    # Check to see if we installed Legate Core
    legate_config = os.path.join(cunumeric_dir, ".legate.core.json")
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
        f = open(
            os.path.join(legate_dir, "share", "legate", ".legate-libs.json"),
            "r",
        )
        libs_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        libs_config = {}
    if openblas_dir is None:
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
            thrust_config = os.path.join(cunumeric_dir, ".thrust.json")
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

    build_cunumeric(
        cunumeric_dir,
        legate_dir,
        openblas_dir,
        cmake,
        cmake_exe,
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
    parser = argparse.ArgumentParser(description="Install cuNumeric.")
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG", "0") == "1",
        help="Build cuNumeric with no optimizations.",
    )
    parser.add_argument(
        "--debug-release",
        dest="debug_release",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG_RELEASE", "0") == "1",
        help="Build cuNumeric with optimizations, but include debugging "
        "symbols.",
    )
    parser.add_argument(
        "--check-bounds",
        dest="check_bounds",
        action="store_true",
        required=False,
        default=False,
        help="Build cuNumeric with bounds checks.",
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
        default=os.environ.get("OPEN_BLAS_DIR"),
        help="Path to OpenBLAS installation directory. Note that providing a "
        "user-defined BLAS library may lead to dynamic library conflicts with "
        "BLAS loaded by Python's Numpy. When using cuNumeric's BLAS, this "
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
        "--cmake",
        action=BooleanFlag,
        default=os.environ.get("USE_CMAKE", "0") == "1",
        help="Build cuNumeric with CMake instead of GNU Make.",
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
        "--clean",
        dest="clean_first",
        action=BooleanFlag,
        default=False,
        help="Clean before build.",
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

    install_cunumeric(unknown=unknown, **vars(args))


if __name__ == "__main__":
    driver()
