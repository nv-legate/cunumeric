#! /usr/bin/env bash

export CFLAGS="-w -fdiagnostics-color=always"
export CXXFLAGS="-w -fdiagnostics-color=always"
export CUDAFLAGS="-w -Xcompiler=-w,-fdiagnostics-color=always"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${JOBS:-${PARALLEL_LEVEL:-$(nproc --ignore=2)}}}"

if [[ -z "$LIBRARY_PATH" ]]; then
    if [[ -d "${CUDA_HOME:-/usr/local/cuda}/lib64/stubs" ]]; then
        export LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64/stubs"
    fi
fi
