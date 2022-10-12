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


def scikit_build_cmake_build_dir(skbuild_dir):
    if os.path.exists(skbuild_dir):
        for f in os.listdir(skbuild_dir):
            if os.path.exists(
                cmake_build := os.path.join(skbuild_dir, f, "cmake-build")
            ):
                return cmake_build
    return None


def find_cmake_val(pattern, filepath):
    return (
        subprocess.check_output(["grep", "--color=never", pattern, filepath])
        .decode("UTF-8")
        .strip()
    )


def was_previously_built_with_different_build_isolation(
    isolated, cunumeric_build_dir
):
    if (
        cunumeric_build_dir is not None
        and os.path.exists(cunumeric_build_dir)
        and os.path.exists(
            cmake_cache := os.path.join(cunumeric_build_dir, "CMakeCache.txt")
        )
    ):
        try:
            if isolated:
                return True
            if find_cmake_val("pip-build-env", cmake_cache):
                return True
        except Exception:
            pass
    return False


def install_cunumeric(
    arch,
    build_isolation,
    check_bounds,
    clean_first,
    cmake_exe,
    cmake_generator,
    conduit,
    cuda_dir,
    cuda,
    curand_dir,
    cutensor_dir,
    debug_release,
    debug,
    editable,
    extra_flags,
    gasnet_dir,
    networks,
    hdf,
    install_dir,
    legate_branch,
    legate_dir,
    legate_url,
    llvm,
    march,
    maxdim,
    maxfields,
    nccl_dir,
    openblas_dir,
    openmp,
    spy,
    tblis_dir,
    thread_count,
    thrust_dir,
    unknown,
    verbose,
):
    if len(networks) > 1:
        print(
            "Warning: Building Realm with multiple networking backends is not "
            "fully supported currently."
        )

    if clean_first is None:
        clean_first = not editable

    print("Verbose build is ", "on" if verbose else "off")
    if verbose:
        print("Options are:")
        print("arch: ", arch)
        print("build_isolation: ", build_isolation)
        print("check_bounds: ", check_bounds)
        print("clean_first: ", clean_first)
        print("cmake_exe: ", cmake_exe)
        print("cmake_generator: ", cmake_generator)
        print("conduit: ", conduit)
        print("cuda_dir: ", cuda_dir)
        print("cuda: ", cuda)
        print("curand_dir: ", curand_dir)
        print("cutensor_dir: ", cutensor_dir)
        print("debug_release: ", debug_release)
        print("debug: ", debug)
        print("editable: ", editable)
        print("extra_flags: ", extra_flags)
        print("gasnet_dir: ", gasnet_dir)
        print("networks: ", networks)
        print("hdf: ", hdf)
        print("install_dir: ", install_dir)
        print("legate_branch: ", legate_branch)
        print("legate_dir: ", legate_dir)
        print("legate_url: ", legate_url)
        print("llvm: ", llvm)
        print("march: ", march)
        print("maxdim: ", maxdim)
        print("maxfields: ", maxfields)
        print("nccl_dir: ", nccl_dir)
        print("openblas_dir: ", openblas_dir)
        print("openmp: ", openmp)
        print("spy: ", spy)
        print("tblis_dir: ", tblis_dir)
        print("thread_count: ", thread_count)
        print("thrust_dir: ", thrust_dir)
        print("unknown: ", unknown)
        print("verbose: ", verbose)

    join = os.path.join
    exists = os.path.exists
    dirname = os.path.dirname
    realpath = os.path.realpath

    cunumeric_dir = dirname(realpath(__file__))

    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    def validate_path(path):
        if path is not None and (path := str(path)) != "":
            if not os.path.isabs(path):
                path = join(cunumeric_dir, path)
            if exists(path := realpath(path)):
                return path
        return None

    cuda_dir = validate_path(cuda_dir)
    nccl_dir = validate_path(nccl_dir)
    tblis_dir = validate_path(tblis_dir)
    legate_dir = validate_path(legate_dir)
    thrust_dir = validate_path(thrust_dir)
    curand_dir = validate_path(curand_dir)
    gasnet_dir = validate_path(gasnet_dir)
    cutensor_dir = validate_path(cutensor_dir)
    openblas_dir = validate_path(openblas_dir)

    if legate_dir is None:
        try:
            import legate.install_info as lg_install_info

            legate_dir = dirname(lg_install_info.libpath)
        except Exception:
            pass

    if verbose:
        print("cuda_dir: ", cuda_dir)
        print("nccl_dir: ", nccl_dir)
        print("tblis_dir: ", tblis_dir)
        print("legate_dir: ", legate_dir)
        print("thrust_dir: ", thrust_dir)
        print("curand_dir: ", curand_dir)
        print("gasnet_dir: ", gasnet_dir)
        print("cutensor_dir: ", cutensor_dir)
        print("openblas_dir: ", openblas_dir)

    skbuild_dir = join(cunumeric_dir, "_skbuild")
    cunumeric_build_dir = scikit_build_cmake_build_dir(skbuild_dir)

    if was_previously_built_with_different_build_isolation(
        build_isolation and not editable, cunumeric_build_dir
    ):
        print("Performing a clean build to accommodate build isolation.")
        clean_first = True

    if clean_first:
        shutil.rmtree(skbuild_dir, ignore_errors=True)
        shutil.rmtree(join(cunumeric_dir, "dist"), ignore_errors=True)
        shutil.rmtree(join(cunumeric_dir, "build"), ignore_errors=True)
        shutil.rmtree(
            join(cunumeric_dir, "cunumeric.egg-info"),
            ignore_errors=True,
        )

    # Configure and build cuNumeric via setup.py
    pip_install_cmd = [sys.executable, "-m", "pip", "install"]
    cmd_env = dict(os.environ.items())

    if unknown is not None:
        try:
            prefix_loc = unknown.index("--prefix")
            prefix_dir = validate_path(unknown[prefix_loc + 1])
            if prefix_dir is not None:
                install_dir = prefix_dir
                unknown = unknown[:prefix_loc] + unknown[prefix_loc + 2 :]
        except Exception:
            pass

    install_dir = validate_path(install_dir)

    if verbose:
        print("install_dir: ", install_dir)

    if install_dir is not None:
        pip_install_cmd += ["--root", "/", "--prefix", str(install_dir)]

    if editable:
        # editable implies build_isolation = False
        pip_install_cmd += ["--no-deps", "--no-build-isolation", "--editable"]
        cmd_env.update({"SETUPTOOLS_ENABLE_FEATURES": "legacy-editable"})
    else:
        if not build_isolation:
            pip_install_cmd += ["--no-deps", "--no-build-isolation"]
        pip_install_cmd += ["--upgrade"]

    pip_install_cmd += ["."]
    if verbose:
        pip_install_cmd += ["-vv"]

    cmake_flags = []

    if cmake_generator:
        cmake_flags += [f"-G'{cmake_generator}'"]

    if debug or verbose:
        cmake_flags += ["--log-level=%s" % ("DEBUG" if debug else "VERBOSE")]

    cmake_flags += f"""\
-DCMAKE_BUILD_TYPE={(
    "Debug" if debug else "RelWithDebInfo" if debug_release else "Release"
)}
-DBUILD_SHARED_LIBS=ON
-DBUILD_MARCH={str(march)}
-DCMAKE_CUDA_ARCHITECTURES={str(arch)}
-DLegion_MAX_DIM={str(maxdim)}
-DLegion_MAX_FIELDS={str(maxfields)}
-DLegion_SPY={("ON" if spy else "OFF")}
-DLegion_BOUNDS_CHECKS={("ON" if check_bounds else "OFF")}
-DLegion_USE_CUDA={("ON" if cuda else "OFF")}
-DLegion_USE_OpenMP={("ON" if openmp else "OFF")}
-DLegion_USE_LLVM={("ON" if llvm else "OFF")}
-DLegion_NETWORKS={";".join(networks)}
-DLegion_USE_HDF5={("ON" if hdf else "OFF")}
""".splitlines()

    if cuda_dir:
        cmake_flags += ["-DCUDAToolkit_ROOT=%s" % cuda_dir]
    if nccl_dir:
        cmake_flags += ["-DNCCL_DIR=%s" % nccl_dir]
    if gasnet_dir:
        cmake_flags += ["-DGASNet_ROOT_DIR=%s" % gasnet_dir]
    if conduit:
        cmake_flags += ["-DGASNet_CONDUIT=%s" % conduit]
    if tblis_dir:
        cmake_flags += ["-Dtblis_ROOT=%s" % tblis_dir]
    if thrust_dir:
        cmake_flags += ["-DThrust_ROOT=%s" % thrust_dir]
    if openblas_dir:
        cmake_flags += ["-DBLAS_DIR=%s" % openblas_dir]
    if cutensor_dir:
        cmake_flags += ["-Dcutensor_DIR=%s" % cutensor_dir]
    # A custom path to cuRAND is ignored when CUDA support is available
    if cuda and curand_dir is not None:
        cmake_flags += ["-Dcunumeric_cuRAND_INCLUDE_DIR=%s" % curand_dir]
    if legate_dir:
        cmake_flags += ["-Dlegate_core_ROOT=%s" % legate_dir]
    if legate_url:
        cmake_flags += ["-Dcunumeric_LEGATE_CORE_REPOSITORY=%s" % legate_url]
    if legate_branch:
        cmake_flags += ["-Dcunumeric_LEGATE_CORE_BRANCH=%s" % legate_branch]

    cmake_flags += extra_flags
    cmd_env.update(
        {
            "SKBUILD_BUILD_OPTIONS": f"-j{str(thread_count)}",
            "SKBUILD_CONFIGURE_OPTIONS": "\n".join(cmake_flags),
        }
    )

    execute_command(pip_install_cmd, verbose, cwd=cunumeric_dir, env=cmd_env)


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
        "--max-dim",
        dest="maxdim",
        type=int,
        default=int(os.environ.get("LEGION_MAX_DIM", 4)),
        help="Maximum number of dimensions that cuNumeric will support",
    )
    parser.add_argument(
        "--max-fields",
        dest="maxfields",
        type=int,
        default=int(os.environ.get("LEGION_MAX_FIELDS", 256)),
        help="Maximum number of fields that cuNumeric will support",
    )
    parser.add_argument(
        "--network",
        dest="networks",
        action="append",
        required=False,
        choices=["gasnet1", "gasnetex", "mpi"],
        default=[],
        help="Realm networking backend to use for multi-node execution.",
    )
    parser.add_argument(
        "--with-gasnet",
        dest="gasnet_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("GASNET"),
        help="Path to GASNet installation directory.",
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
        default="branch-22.10",
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
        "--with-curand",
        dest="curand_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("CURAND_PATH"),
        help="Path to cuRAND installation directory. This flag is ignored "
        "if Legate Core was built with CUDA support.",
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
        default=(None if shutil.which("ninja") is None else "Ninja"),
        choices=["Ninja", "Unix Makefiles", None],
        help="The CMake makefiles generator",
    )
    parser.add_argument(
        "--cuda",
        action=BooleanFlag,
        default=os.environ.get("USE_CUDA", "0") == "1",
        help="Build cuNumeric with CUDA support.",
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
        "--openmp",
        action=BooleanFlag,
        default=os.environ.get("USE_OPENMP", "0") == "1",
        help="Build cuNumeric with OpenMP support.",
    )
    parser.add_argument(
        "--march",
        dest="march",
        required=False,
        default="native",
        help="Specify the target CPU architecture.",
    )
    parser.add_argument(
        "--llvm",
        dest="llvm",
        action="store_true",
        required=False,
        default=os.environ.get("USE_LLVM", "0") == "1",
        help="Build cuNumeric with LLVM support.",
    )
    parser.add_argument(
        "--hdf5",
        "--hdf",
        dest="hdf",
        action="store_true",
        required=False,
        default=os.environ.get("USE_HDF", "0") == "1",
        help="Build cuNumeric with HDF support.",
    )
    parser.add_argument(
        "--spy",
        dest="spy",
        action="store_true",
        required=False,
        default=os.environ.get("USE_SPY", "0") == "1",
        help="Build cuNumeric with detailed Legion Spy enabled.",
    )
    parser.add_argument(
        "--conduit",
        dest="conduit",
        action="store",
        required=False,
        # TODO: To support UDP conduit, we would need to add a special case on
        # the legate launcher.
        # See https://github.com/nv-legate/legate.core/issues/294.
        choices=["ibv", "ucx", "aries", "mpi"],
        default=os.environ.get("CONDUIT"),
        help="Build cuNumeric with specified GASNet conduit.",
    )
    parser.add_argument(
        "--clean",
        dest="clean_first",
        action=BooleanFlag,
        default=None,
        help="Clean before build.",
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
        default=os.environ.get("CPU_COUNT"),
        help="Number of threads used to compile.",
    )
    parser.add_argument(
        "--editable",
        dest="editable",
        action="store_true",
        required=False,
        default=False,
        help="Perform an editable install. Disables --build-isolation if set "
        "(passing --no-deps --no-build-isolation to pip).",
    )
    parser.add_argument(
        "--build-isolation",
        dest="build_isolation",
        action=BooleanFlag,
        required=False,
        default=True,
        help="Enable isolation when building a modern source distribution. "
        "Build dependencies specified by PEP 518 must be already "
        "installed if this option is used.",
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
