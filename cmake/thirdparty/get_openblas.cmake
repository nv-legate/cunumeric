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

function(find_or_configure_OpenBLAS)
  set(oneValueArgs VERSION REPOSITORY BRANCH PINNED_TAG EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(INTERFACE64 OFF)
  set(BLAS_name "OpenBLAS")
  set(BLAS_target "openblas")

  # cuNumeric presently requires OpenBLAS
  set(BLA_VENDOR OpenBLAS)

  # TODO: should we find (or build) 64-bit BLAS?
  if(FALSE AND (CMAKE_SIZEOF_VOID_P EQUAL 8))
    set(INTERFACE64 ON)
    set(BLAS_name "OpenBLAS64")
    set(BLAS_target "openblas_64")
    set(BLA_SIZEOF_INTEGER 8)
  endif()

  set(FIND_PKG_ARGS      ${PKG_VERSION}
      GLOBAL_TARGETS     ${BLAS_target}
      BUILD_EXPORT_SET   cunumeric-exports
      INSTALL_EXPORT_SET cunumeric-exports)

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cpm_helpers.cmake)
  if(PKG_BRANCH)
    get_cpm_git_args(BLAS_cpm_git_args REPOSITORY ${PKG_REPOSITORY} BRANCH ${PKG_BRANCH})
  else()
    get_cpm_git_args(BLAS_cpm_git_args REPOSITORY ${PKG_REPOSITORY} TAG ${PKG_PINNED_TAG})
  endif()

  cmake_policy(GET CMP0048 CMP0048_orig)
  cmake_policy(GET CMP0054 CMP0054_orig)
  set(CMAKE_POLICY_DEFAULT_CMP0048 OLD)
  set(CMAKE_POLICY_DEFAULT_CMP0054 NEW)

  rapids_cpm_find(BLAS ${FIND_PKG_ARGS}
      CPM_ARGS
        ${BLAS_cpm_git_args}
        EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
        OPTIONS "USE_CUDA 0"
                "C_LAPACK ON"
                "USE_THREAD ON"
                "NUM_PARALLEL 32"
                "BUILD_TESTING OFF"
                "BUILD_WITHOUT_CBLAS OFF"
                "BUILD_WITHOUT_LAPACK OFF"
                "INTERFACE64 ${INTERFACE64}"
                "USE_OPENMP ${Legion_USE_OpenMP}")

  set(CMAKE_POLICY_DEFAULT_CMP0048 ${CMP0048_orig})
  set(CMAKE_POLICY_DEFAULT_CMP0054 ${CMP0054_orig})

  if(BLAS_ADDED AND (TARGET ${BLAS_target}))

    # Ensure we export the name of the actual target, not an alias target
    get_target_property(BLAS_aliased_target ${BLAS_target} ALIASED_TARGET)
    if(TARGET ${BLAS_aliased_target})
      set(BLAS_target ${BLAS_aliased_target})
    endif()
    # Make an BLAS::BLAS alias target
    if(NOT TARGET BLAS::BLAS)
      add_library(BLAS::BLAS ALIAS ${BLAS_target})
    endif()

    # Set build INTERFACE_INCLUDE_DIRECTORIES appropriately
    get_target_property(BLAS_include_dirs ${BLAS_target} INCLUDE_DIRECTORIES)
    target_include_directories(${BLAS_target}
        PUBLIC $<BUILD_INTERFACE:${BLAS_BINARY_DIR}>
               # lapack[e] etc. include paths
               $<BUILD_INTERFACE:${BLAS_include_dirs}>
               # contains openblas_config.h
               $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
               # contains cblas.h and f77blas.h
               $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
               )

    string(JOIN "\n" code_string
      "if(NOT TARGET BLAS::BLAS)"
      "  add_library(BLAS::BLAS ALIAS ${BLAS_target})"
      "endif()"
    )

    # Generate openblas-config.cmake in build dir
    rapids_export(BUILD BLAS
      VERSION ${PKG_VERSION}
      EXPORT_SET "${BLAS_name}Targets"
      GLOBAL_TARGETS ${BLAS_target}
      FINAL_CODE_BLOCK code_string)

    # Do `CPMFindPackage(BLAS)` in build dir
    rapids_export_package(BUILD BLAS cunumeric-exports
      VERSION ${PKG_VERSION} GLOBAL_TARGETS ${BLAS_target})

    # Tell cmake where it can find the generated blas-config.cmake
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD BLAS [=[${CMAKE_CURRENT_LIST_DIR}]=] cunumeric-exports)
  endif()
endfunction()

if(NOT DEFINED cunumeric_OPENBLAS_VERSION)
  # Before v0.3.18, OpenBLAS's throws CMake errors when configuring
  set(cunumeric_OPENBLAS_VERSION "0.3.20")
endif()

if(NOT DEFINED cunumeric_OPENBLAS_BRANCH)
  set(cunumeric_OPENBLAS_BRANCH "")
endif()

if(NOT DEFINED cunumeric_OPENBLAS_TAG)
  set(cunumeric_OPENBLAS_TAG v${cunumeric_OPENBLAS_VERSION})
endif()

if(NOT DEFINED cunumeric_OPENBLAS_REPOSITORY)
  set(cunumeric_OPENBLAS_REPOSITORY https://github.com/xianyi/OpenBLAS.git)
endif()

find_or_configure_OpenBLAS(VERSION          ${cunumeric_OPENBLAS_VERSION}
                           REPOSITORY       ${cunumeric_OPENBLAS_REPOSITORY}
                           BRANCH           ${cunumeric_OPENBLAS_BRANCH}
                           PINNED_TAG       ${cunumeric_OPENBLAS_TAG}
                           EXCLUDE_FROM_ALL ${cunumeric_EXCLUDE_OPENBLAS_FROM_ALL}
)
