#=============================================================================
# Copyright 2022 NVIDIA Corporation
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
#=============================================================================

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

cmake_policy(SET CMP0077 NEW)

##############################################################################
# - Download and initialize RAPIDS CMake helpers -----------------------------

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.06/RAPIDS.cmake
    ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build cuNumeric shared libraries" ON)
option(CUNUMERIC_CHECK_BOUNDS "Build cuNumeric with bounds checks (expensive)" OFF)
option(CUNUMERIC_EXCLUDE_TBLIS_FROM_ALL "Exclude tblis targets from legate.core's 'all' target" OFF)
option(CUNUMERIC_EXCLUDE_LEGATE_CORE_FROM_ALL "Exclude legate.core targets from legate.core's 'all' target" OFF)

# Needs to run before `project` call
rapids_cuda_init_architectures(CUNUMERIC)

project(CUNUMERIC VERSION 22.05.00 LANGUAGES C CXX CUDA)

# Write the version header
rapids_cmake_write_version_file(include/cunumeric/version_config.hpp)

##############################################################################
# - Build Type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# Needed to integrate with LLVM/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - Dependencies -------------------------------------------------------------
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET cunumeric-exports
  INSTALL_EXPORT_SET cunumeric-exports
)

# add third party dependencies using CPM
rapids_cpm_init()

find_package(OpenMP)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

include(cmake/thirdparty/get_nccl.cmake)
include(cmake/thirdparty/get_tblis.cmake)
include(cmake/thirdparty/get_cutensor.cmake)
include(cmake/thirdparty/get_legate_core.cmake)

##############################################################################
# - cuNumeric ----------------------------------------------------------------

set(CUNUMERIC_SOURCES "")
set(CUNUMERIC_CXX_DEFS "")
set(CUNUMERIC_CUDA_DEFS "")
set(CUNUMERIC_CXX_OPTIONS "")
set(CUNUMERIC_CUDA_OPTIONS "")

# Add `src/cunumeric.mk` sources
list(APPEND CUNUMERIC_SOURCES
  src/cunumeric/ternary/where.cc
  src/cunumeric/binary/binary_op.cc
  src/cunumeric/binary/binary_red.cc
  src/cunumeric/unary/scalar_unary_red.cc
  src/cunumeric/unary/unary_op.cc
  src/cunumeric/unary/unary_red.cc
  src/cunumeric/unary/convert.cc
  src/cunumeric/nullary/arange.cc
  src/cunumeric/nullary/eye.cc
  src/cunumeric/nullary/fill.cc
  src/cunumeric/nullary/window.cc
  src/cunumeric/index/advanced_indexing.cc
  src/cunumeric/index/choose.cc
  src/cunumeric/index/repeat.cc
  src/cunumeric/index/zip.cc
  src/cunumeric/item/read.cc
  src/cunumeric/item/write.cc
  src/cunumeric/matrix/contract.cc
  src/cunumeric/matrix/diag.cc
  src/cunumeric/matrix/gemm.cc
  src/cunumeric/matrix/matmul.cc
  src/cunumeric/matrix/matvecmul.cc
  src/cunumeric/matrix/dot.cc
  src/cunumeric/matrix/potrf.cc
  src/cunumeric/matrix/syrk.cc
  src/cunumeric/matrix/tile.cc
  src/cunumeric/matrix/transpose.cc
  src/cunumeric/matrix/trilu.cc
  src/cunumeric/matrix/trsm.cc
  src/cunumeric/matrix/util.cc
  src/cunumeric/random/rand.cc
  src/cunumeric/search/nonzero.cc
  src/cunumeric/set/unique.cc
  src/cunumeric/set/unique_reduce.cc
  src/cunumeric/stat/bincount.cc
  src/cunumeric/convolution/convolve.cc
  src/cunumeric/transform/flip.cc
  src/cunumeric/arg.cc
  src/cunumeric/mapper.cc
  src/cunumeric/cephes/chbevl.cc
  src/cunumeric/cephes/i0.cc
)

if(OpenMP_FOUND)
  list(APPEND CUNUMERIC_SOURCES
    src/cunumeric/ternary/where_omp.cc
    src/cunumeric/binary/binary_op_omp.cc
    src/cunumeric/binary/binary_red_omp.cc
    src/cunumeric/unary/unary_op_omp.cc
    src/cunumeric/unary/scalar_unary_red_omp.cc
    src/cunumeric/unary/unary_red_omp.cc
    src/cunumeric/unary/convert_omp.cc
    src/cunumeric/nullary/arange_omp.cc
    src/cunumeric/nullary/eye_omp.cc
    src/cunumeric/nullary/fill_omp.cc
    src/cunumeric/nullary/window_omp.cc
    src/cunumeric/index/advanced_indexing_omp.cc
    src/cunumeric/index/choose_omp.cc
    src/cunumeric/index/repeat_omp.cc
    src/cunumeric/index/zip_omp.cc
    src/cunumeric/matrix/contract_omp.cc
    src/cunumeric/matrix/diag_omp.cc
    src/cunumeric/matrix/gemm_omp.cc
    src/cunumeric/matrix/matmul_omp.cc
    src/cunumeric/matrix/matvecmul_omp.cc
    src/cunumeric/matrix/dot_omp.cc
    src/cunumeric/matrix/potrf_omp.cc
    src/cunumeric/matrix/syrk_omp.cc
    src/cunumeric/matrix/tile_omp.cc
    src/cunumeric/matrix/transpose_omp.cc
    src/cunumeric/matrix/trilu_omp.cc
    src/cunumeric/matrix/trsm_omp.cc
    src/cunumeric/matrix/util_omp.cc
    src/cunumeric/random/rand_omp.cc
    src/cunumeric/search/nonzero_omp.cc
    src/cunumeric/set/unique_omp.cc
    src/cunumeric/stat/bincount_omp.cc
    src/cunumeric/convolution/convolve_omp.cc
    src/cunumeric/transform/flip_omp.cc
  )
endif()

list(APPEND CUNUMERIC_SOURCES
  src/cunumeric/ternary/where.cu
  src/cunumeric/binary/binary_op.cu
  src/cunumeric/binary/binary_red.cu
  src/cunumeric/unary/scalar_unary_red.cu
  src/cunumeric/unary/unary_red.cu
  src/cunumeric/unary/unary_op.cu
  src/cunumeric/unary/convert.cu
  src/cunumeric/nullary/arange.cu
  src/cunumeric/nullary/eye.cu
  src/cunumeric/nullary/fill.cu
  src/cunumeric/nullary/window.cu
  src/cunumeric/index/advanced_indexing.cu
  src/cunumeric/index/choose.cu
  src/cunumeric/index/repeat.cu
  src/cunumeric/index/zip.cu
  src/cunumeric/item/read.cu
  src/cunumeric/item/write.cu
  src/cunumeric/matrix/contract.cu
  src/cunumeric/matrix/diag.cu
  src/cunumeric/matrix/gemm.cu
  src/cunumeric/matrix/matmul.cu
  src/cunumeric/matrix/matvecmul.cu
  src/cunumeric/matrix/dot.cu
  src/cunumeric/matrix/potrf.cu
  src/cunumeric/matrix/syrk.cu
  src/cunumeric/matrix/tile.cu
  src/cunumeric/matrix/transpose.cu
  src/cunumeric/matrix/trilu.cu
  src/cunumeric/matrix/trsm.cu
  src/cunumeric/random/rand.cu
  src/cunumeric/search/nonzero.cu
  src/cunumeric/set/unique.cu
  src/cunumeric/stat/bincount.cu
  src/cunumeric/convolution/convolve.cu
  src/cunumeric/fft/fft.cu
  src/cunumeric/transform/flip.cu
  src/cunumeric/cudalibs.cu
  src/cunumeric/cunumeric.cu
)

# Add `src/cunumeric/sort/sort.mk` sources
list(APPEND CUNUMERIC_SOURCES
  src/cunumeric/sort/sort.cc
)

if(OpenMP_FOUND)
  list(APPEND CUNUMERIC_SOURCES
    src/cunumeric/sort/sort_omp.cc
  )
endif()

list(APPEND CUNUMERIC_SOURCES
  src/cunumeric/sort/sort.cu
  src/cunumeric/sort/cub_sort_bool.cu
  src/cunumeric/sort/cub_sort_int8.cu
  src/cunumeric/sort/cub_sort_int16.cu
  src/cunumeric/sort/cub_sort_int32.cu
  src/cunumeric/sort/cub_sort_int64.cu
  src/cunumeric/sort/cub_sort_uint8.cu
  src/cunumeric/sort/cub_sort_uint16.cu
  src/cunumeric/sort/cub_sort_uint32.cu
  src/cunumeric/sort/cub_sort_uint64.cu
  src/cunumeric/sort/cub_sort_half.cu
  src/cunumeric/sort/cub_sort_float.cu
  src/cunumeric/sort/cub_sort_double.cu
  src/cunumeric/sort/thrust_sort_bool.cu
  src/cunumeric/sort/thrust_sort_int8.cu
  src/cunumeric/sort/thrust_sort_int16.cu
  src/cunumeric/sort/thrust_sort_int32.cu
  src/cunumeric/sort/thrust_sort_int64.cu
  src/cunumeric/sort/thrust_sort_uint8.cu
  src/cunumeric/sort/thrust_sort_uint16.cu
  src/cunumeric/sort/thrust_sort_uint32.cu
  src/cunumeric/sort/thrust_sort_uint64.cu
  src/cunumeric/sort/thrust_sort_half.cu
  src/cunumeric/sort/thrust_sort_float.cu
  src/cunumeric/sort/thrust_sort_double.cu
  src/cunumeric/sort/thrust_sort_complex64.cu
  src/cunumeric/sort/thrust_sort_complex128.cu
)

list(APPEND CUNUMERIC_SOURCES
  # This must always be the last file!
  # It guarantees we do our registration callback
  # only after all task variants are recorded
  src/cunumeric/cunumeric.cc
)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  list(APPEND CUNUMERIC_CXX_DEFS DEBUG_CUNUMERIC)
  list(APPEND CUNUMERIC_CUDA_DEFS DEBUG_CUNUMERIC)
endif()

if(CUNUMERIC_CHECK_BOUNDS)
  list(APPEND CUNUMERIC_CXX_DEFS BOUNDS_CHECKS)
endif()

list(APPEND CUNUMERIC_CUDA_OPTIONS -Xfatbin=-compress-all)
list(APPEND CUNUMERIC_CUDA_OPTIONS --expt-extended-lambda)
list(APPEND CUNUMERIC_CUDA_OPTIONS --expt-relaxed-constexpr)
list(APPEND CUNUMERIC_CXX_OPTIONS -Wno-deprecated-declarations)
list(APPEND CUNUMERIC_CUDA_OPTIONS -Wno-deprecated-declarations)

add_library(cunumeric ${CUNUMERIC_SOURCES})
add_library(cunumeric::cunumeric ALIAS cunumeric)

set_target_properties(cunumeric
           PROPERTIES BUILD_RPATH                         "\$ORIGIN"
                      INSTALL_RPATH                       "\$ORIGIN"
                      CXX_STANDARD                        17
                      CXX_STANDARD_REQUIRED               ON
                      POSITION_INDEPENDENT_CODE           ON
                      INTERFACE_POSITION_INDEPENDENT_CODE ON
                      CUDA_STANDARD                       17
                      CUDA_STANDARD_REQUIRED              ON)

target_link_libraries(cunumeric
  PRIVATE NCCL::NCCL
          BLAS::BLAS
          LAPACK::LAPACK
          $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
          tblis::tblis
          cutensor::cutensor
          CUDA::nvToolsExt
          legate::core)

# Change THRUST_DEVICE_SYSTEM for `.cpp` files
if(OpenMP_FOUND)
  list(APPEND CUNUMERIC_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND CUNUMERIC_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
endif()

###
# Preinclude `realm_defines.h` and `legion_defines.h` in all compilations.
# TODO: It isn't possible to expose this as part of the public legate_core
# target's interface in a portable way, so this code needs to be duplicated
# in cuNumeric.
#
# A better solution would be to add #include directives to the requisite
# legate.core and cuNumeric headers.
###

get_target_property(realm_includes Legion::RealmRuntime INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(legion_includes Legion::LegionRuntime INTERFACE_INCLUDE_DIRECTORIES)

set(extra_includes "")
foreach(_dir IN LISTS realm_includes legion_includes)
  if(EXISTS "${_dir}/realm_defines.h")
    list(APPEND extra_includes "${_dir}/realm_defines.h")
  endif()
  if(EXISTS "${_dir}/legion_defines.h")
    list(APPEND extra_includes "${_dir}/legion_defines.h")
  endif()
endforeach()

list(REMOVE_DUPLICATES extra_includes)
list(LENGTH extra_includes num_extra_includes)
if(num_extra_includes GREATER 2)
  list(SORT extra_includes)
  list(SUBLIST extra_includes 0 2 extra_includes)
endif()
list(TRANSFORM extra_includes PREPEND "SHELL:-include ")

target_compile_options(cunumeric
  PRIVATE ${extra_includes}
          "$<$<COMPILE_LANGUAGE:CXX>:${CUNUMERIC_CXX_OPTIONS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${CUNUMERIC_CUDA_OPTIONS}>")

target_compile_definitions(cunumeric
  PUBLIC  "$<$<COMPILE_LANGUAGE:CXX>:${CUNUMERIC_CXX_DEFS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${CUNUMERIC_CUDA_DEFS}>")

target_include_directories(cunumeric
  PRIVATE
    $<BUILD_INTERFACE:${CUNUMERIC_SOURCE_DIR}/src>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
[=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=])

# ensure CUDA symbols aren't relocated to the middle of the debug build binaries
target_link_options(cunumeric PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")

##############################################################################
# - install targets-----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS cunumeric
        DESTINATION ${lib_dir}
        EXPORT cunumeric-exports)

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide targets for cuNumeric, an Aspiring Drop-In Replacement for NumPy at Scale.

Imported Targets:
  - cunumeric::cunumeric

]=])

set(code_string "")

rapids_export(
  INSTALL cunumeric
  EXPORT_SET cunumeric-exports
  GLOBAL_TARGETS cunumeric
  NAMESPACE cunumeric::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD cunumeric
  EXPORT_SET cunumeric-exports
  GLOBAL_TARGETS cunumeric
  NAMESPACE cunumeric::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)
