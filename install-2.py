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
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys

import setuptools

# Flush output on newlines
sys.stdout.reconfigure(line_buffering=True)

os_name = platform.system()

if os_name == "Linux":
    pass
elif os_name == "Darwin":
    pass
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


def execute_command(args, verbose, **kwargs):
    if verbose:
        print('Executing: "', " ".join(args), '" with ', kwargs)
    subprocess.check_call(args, **kwargs)


def cmake_build(
    cmake_exe,
    build_dir,
    thread_count,
    verbose,
):
    cmake_flags = ["--build", build_dir]
    if verbose:
        cmake_flags += ["-v"]
    if thread_count is not None:
        cmake_flags += ["-j", str(thread_count)]

    execute_command([cmake_exe] + cmake_flags, verbose)


def cmake_install(
    cmake_exe,
    build_dir,
    verbose,
    install_dir=None,
):
    cmake_flags = ["--install", build_dir]

    if install_dir is not None:
        cmake_flags += ["--prefix", install_dir]

    execute_command([cmake_exe] + cmake_flags, verbose)


def configure_cunumeric_cpp(
    cunumeric_dir,
    build_dir,
    legate_dir,
    legate_url,
    legate_branch,
    openblas_dir,
    tblis_dir,
    cutensor_dir,
    nccl_dir,
    thrust_dir,
    arch,
    march,
    cuda,
    cuda_dir,
    cmake_exe,
    cmake_generator,
    debug,
    debug_release,
    check_bounds,
    verbose,
    extra_flags,
):
    cmake_flags = [
        "-G",
        cmake_generator,
        "-S",
        cunumeric_dir,
        "-B",
        build_dir,
    ]

    if debug or verbose:
        cmake_flags += ["--log-level=%s" % ("DEBUG" if debug else "VERBOSE")]

    cmake_flags += [
        "-DCMAKE_BUILD_TYPE=%s"
        % (
            "Debug"
            if debug
            else "RelWithDebInfo"
            if debug_release
            else "Release"
        ),
        "-DBUILD_SHARED_LIBS=ON",
        "-DBUILD_MARCH=%s" % march,
        "-DCMAKE_CUDA_ARCHITECTURES=%s" % arch,
        "-DLegion_USE_CUDA=%s" % ("ON" if cuda else "OFF"),
        "-DLegion_BOUNDS_CHECKS=%s" % ("ON" if check_bounds else "OFF"),
    ]

    if cuda_dir:
        cmake_flags += ["-DCUDAToolkit_ROOT=%s" % cuda_dir]
    if nccl_dir:
        cmake_flags += ["-DNCCL_DIR=%s" % nccl_dir]
    if tblis_dir:
        cmake_flags += ["-DTBLIS_DIR=%s" % tblis_dir]
    if thrust_dir:
        cmake_flags += ["-DThrust_ROOT=%s" % thrust_dir]
    if openblas_dir:
        cmake_flags += ["-DBLAS_DIR=%s" % openblas_dir]
    if cutensor_dir:
        cmake_flags += ["-Dcutensor_DIR=%s" % cutensor_dir]
    if legate_dir:
        cmake_flags += ["-Dlegate_core_ROOT=%s" % legate_dir]
    if legate_url:
        cmake_flags += ["-DCUNUMERIC_LEGATE_CORE_REPOSITORY=%s" % legate_url]
    if legate_branch:
        cmake_flags += ["-DCUNUMERIC_LEGATE_CORE_BRANCH=%s" % legate_branch]

    execute_command(
        [cmake_exe] + cmake_flags + extra_flags, verbose, cwd=cunumeric_dir
    )


def build_cunumeric_python(
    cunumeric_dir,
    build_dir,
    verbose,
    unknown,
):
    cmd = [
        sys.executable,
        "setup.py",
        "install",
        "--recurse",
        "--prefix",
        build_dir,
    ]

    # Work around breaking change in setuptools 60
    if int(setuptools.__version__.split(".")[0]) >= 60:
        cmd += ["--single-version-externally-managed", "--root=/"]

    if unknown is not None:
        if "--prefix" in unknown:
            raise Exception(
                "cuNumeric cannot be installed in a different location than "
                "Legate Core, please remove the --prefix argument"
            )
        cmd += unknown

    execute_command(cmd, verbose, cwd=cunumeric_dir)


def install_cunumeric(
    arch,
    march,
    cuda,
    cuda_dir,
    cmake_exe,
    cmake_generator,
    install_dir,
    legate_dir,
    legate_url,
    legate_branch,
    openblas_dir,
    tblis_dir,
    cutensor_dir,
    thrust_dir,
    nccl_dir,
    debug,
    debug_release,
    check_bounds,
    clean_first,
    python_only,
    thread_count,
    verbose,
    extra_flags,
    unknown,
):
    print("Verbose build is ", "on" if verbose else "off")
    if verbose:
        print("Options are:")
        print("arch: ", arch)
        print("march: ", march)
        print("cuda: ", cuda)
        print("cuda_dir: ", cuda_dir)
        print("cmake_exe: ", cmake_exe)
        print("cmake_generator: ", cmake_generator)
        print("legate_dir: ", legate_dir)
        print("legate_url: ", legate_url)
        print("legate_branch: ", legate_branch)
        print("openblas_dir: ", openblas_dir)
        print("tblis_dir: ", tblis_dir)
        print("cutensor_dir: ", cutensor_dir)
        print("thrust_dir: ", thrust_dir)
        print("nccl_dir: ", nccl_dir)
        print("debug: ", debug)
        print("debug_release: ", debug_release)
        print("check_bounds: ", check_bounds)
        print("clean_first: ", clean_first)
        print("python_only: ", python_only)
        print("thread_count: ", thread_count)
        print("verbose: ", verbose)
        print("unknown: ", unknown)

    cunumeric_dir = os.path.dirname(os.path.realpath(__file__))

    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    def validate_path(path):
        if path is not None:
            path = os.path.realpath(path)
            if not os.path.exists(path):
                path = None
        return path

    cuda_dir = validate_path(cuda_dir)
    tblis_dir = validate_path(tblis_dir)
    legate_dir = validate_path(legate_dir)
    thrust_dir = validate_path(thrust_dir)
    cutensor_dir = validate_path(cutensor_dir)
    openblas_dir = validate_path(openblas_dir)

    if install_dir is None:
        if legate_dir is not None:
            install_dir = os.path.join(legate_dir, "install")
        else:
            install_dir = os.path.join(cunumeric_dir, "install")
    else:
        install_dir = os.path.realpath(install_dir)

    if not python_only:
        build_dir = os.path.join(cunumeric_dir, "build")

        if clean_first:
            shutil.rmtree(build_dir, ignore_errors=True)

        # Configure cuNumeric C++
        configure_cunumeric_cpp(
            cunumeric_dir,
            build_dir,
            legate_dir,
            legate_url,
            legate_branch,
            openblas_dir,
            tblis_dir,
            cutensor_dir,
            nccl_dir,
            thrust_dir,
            arch,
            march,
            cuda,
            cuda_dir,
            cmake_exe,
            cmake_generator,
            debug,
            debug_release,
            check_bounds,
            verbose,
            extra_flags,
        )

        # Build cuNumeric C++
        cmake_build(
            cmake_exe,
            build_dir,
            thread_count,
            verbose,
        )

        # Install cuNumeric C++
        cmake_install(cmake_exe, build_dir, verbose, install_dir)

    build_cunumeric_python(cunumeric_dir, build_dir, verbose, unknown)


def driver():
    parser = argparse.ArgumentParser(description="Install cuNumeric.")
    parser.add_argument(
        "--install-dir",
        dest="install_dir",
        metavar="DIR",
        required=False,
        default=None,
        help="Path to install cuNumeric software",
    )
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
        default=os.environ.get("LEGATE_DIR"),
        help="Path to Legate Core installation directory.",
    )
    parser.add_argument(
        "--legate-url",
        dest="legate_url",
        required=False,
        default="https://github.com/nv-legate/legate.core.git",
        help="Legate git URL to build cuNumeric with.",
    )
    parser.add_argument(
        "--legate-branch",
        dest="legate_branch",
        required=False,
        default="control_replication",
        help="Legate branch to build cuNumeric with.",
    )
    parser.add_argument(
        "--with-openblas",
        dest="openblas_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("OPENBLAS_PATH"),
        help="Path to OpenBLAS installation directory. Note that providing a "
        "user-defined BLAS library may lead to dynamic library conflicts with "
        "BLAS loaded by Python's Numpy. When using cuNumeric's BLAS, this "
        "issue is prevented by a custom library name.",
    )
    parser.add_argument(
        "--with-tblis",
        dest="tblis_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("TBLIS_PATH"),
        help="Path to TBLIS installation directory.",
    )
    parser.add_argument(
        "--with-cutensor",
        dest="cutensor_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("CUTENSOR_PATH"),
        help="Path to cuTensor installation directory.",
    )
    parser.add_argument(
        "--with-thrust",
        dest="thrust_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("THRUST_PATH"),
        help="Path to Thrust installation directory.",
    )
    parser.add_argument(
        "--with-nccl",
        dest="nccl_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("NCCL_PATH"),
        help="Path to NCCL installation directory.",
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
        "--cmake-generator",
        dest="cmake_generator",
        required=False,
        default="Unix Makefiles",
        choices=["Unix Makefiles", "Ninja"],
        help="The CMake makefiles generator",
    )
    parser.add_argument(
        "--cuda",
        action=BooleanFlag,
        default=os.environ.get("USE_CUDA", "0") == "1",
        help="Build Legate with CUDA support.",
    )
    parser.add_argument(
        "--with-cuda",
        dest="cuda_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("CUDA"),
        help="Path to CUDA installation directory.",
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        action="store",
        required=False,
        default="NATIVE",
        help="Specify the target GPU architecture.",
    )
    parser.add_argument(
        "--march",
        dest="march",
        required=False,
        default="native",
        help="Specify the target CPU architecture.",
    )
    parser.add_argument(
        "--clean",
        dest="clean_first",
        action=BooleanFlag,
        default=True,
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
        "--extra",
        dest="extra_flags",
        action="append",
        required=False,
        default=[],
        help="Extra CMake flags.",
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
