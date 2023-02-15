#!/bin/bash
  
# Rewrite conda's -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY to
#                 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH
CMAKE_ARGS="$(echo "$CMAKE_ARGS" | sed -r "s@_INCLUDE=ONLY@_INCLUDE=BOTH@g")"

# Add our options to conda's CMAKE_ARGS
CMAKE_ARGS+="
--log-level=VERBOSE"

# We rely on an environment variable to determine if we need to build cpu-only bits
if [ -z "$CPU_ONLY" ]; then
  # cutensor, relying on the conda cutensor package
  CMAKE_ARGS+="
-Dcutensor_DIR=$PREFIX
-DCMAKE_CUDA_ARCHITECTURES:LIST=60-real;70-real;75-real;80-real;90
-DBUILD_MARCH=haswell
"
else
  # When we build without cuda, we need to provide the location of curand
  CMAKE_ARGS+="
-Dcunumeric_cuRAND_INCLUDE_DIR=$PREFIX
"
fi

# Do not compile with NDEBUG until Legion handles it without warnings
export CFLAGS="-UNDEBUG"
export CXXFLAGS="-UNDEBUG"
export CPPFLAGS="-UNDEBUG"
export CUDAFLAGS="-UNDEBUG"
export CMAKE_GENERATOR=Ninja
export CUDAHOSTCXX=${CXX}

echo "Build starting on $(date)"

cmake -S . -B build ${CMAKE_ARGS} -DCMAKE_BUILD_PARALLEL_LEVEL=$CPU_COUNT
cmake --build build -j$CPU_COUNT
cmake --install build

CMAKE_ARGS="
-DFIND_CUNUMERIC_CPP=ON
-Dcunumeric_ROOT=$PREFIX
"

SKBUILD_BUILD_OPTIONS=-j$CPU_COUNT \
$PYTHON -m pip install             \
  --root /                         \
  --no-deps                        \
  --prefix "$PREFIX"               \
  --no-build-isolation             \
  --cache-dir "$PIP_CACHE_DIR"     \
  --disable-pip-version-check      \
  . -vv

echo "Build ending on $(date)"

# Legion leaves an egg-info file which will confuse conda trying to pick up the information
# Remove it so the legate-core is the only egg-info file added
rm -rf $SP_DIR/legion*egg-info
