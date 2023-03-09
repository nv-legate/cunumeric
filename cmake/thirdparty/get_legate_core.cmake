#=============================================================================
# Copyright 2022-2023 NVIDIA Corporation
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

function(find_or_configure_legate_core)
  set(oneValueArgs VERSION REPOSITORY BRANCH EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${PKG_VERSION} legate_core PKG_VERSION)

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(legate_core version git_repo git_branch shallow exclude_from_all)

  set(version ${PKG_VERSION})
  set(exclude_from_all ${PKG_EXCLUDE_FROM_ALL})
  if(PKG_BRANCH)
    set(git_branch "${PKG_BRANCH}")
  endif()
  if(PKG_REPOSITORY)
    set(git_repo "${PKG_REPOSITORY}")
  endif()

  set(FIND_PKG_ARGS
      GLOBAL_TARGETS     legate::core
      BUILD_EXPORT_SET   cunumeric-exports
      INSTALL_EXPORT_SET cunumeric-exports)

  # First try to find legate_core via find_package()
  # so the `Legion_USE_*` variables are visible
  # Use QUIET find by default.
  set(_find_mode QUIET)
  # If legate_core_DIR/legate_core_ROOT are defined as something other than empty or NOTFOUND
  # use a REQUIRED find so that the build does not silently download legate.core.
  if(legate_core_DIR OR legate_core_ROOT)
    set(_find_mode REQUIRED)
  endif()
  rapids_find_package(legate_core ${version} EXACT CONFIG ${_find_mode} ${FIND_PKG_ARGS})

  if(legate_core_FOUND)
    message(STATUS "CPM: using local package legate_core@${version}")
  else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cpm_helpers.cmake)
    get_cpm_git_args(legate_core_cpm_git_args REPOSITORY ${git_repo} BRANCH ${git_branch})

    message(VERBOSE "cunumeric: legate.core version: ${version}")
    message(VERBOSE "cunumeric: legate.core git_repo: ${git_repo}")
    message(VERBOSE "cunumeric: legate.core git_branch: ${git_branch}")
    message(VERBOSE "cunumeric: legate.core exclude_from_all: ${exclude_from_all}")

    rapids_cpm_find(legate_core ${version} ${FIND_PKG_ARGS}
        CPM_ARGS
          ${legate_core_cpm_git_args}
          FIND_PACKAGE_ARGUMENTS EXACT
          EXCLUDE_FROM_ALL       ${exclude_from_all}
    )
  endif()

  set(Legion_USE_CUDA ${Legion_USE_CUDA} PARENT_SCOPE)
  set(Legion_USE_OpenMP ${Legion_USE_OpenMP} PARENT_SCOPE)
  set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS} PARENT_SCOPE)

  message(VERBOSE "Legion_USE_CUDA=${Legion_USE_CUDA}")
  message(VERBOSE "Legion_USE_OpenMP=${Legion_USE_OpenMP}")
  message(VERBOSE "Legion_BOUNDS_CHECKS=${Legion_BOUNDS_CHECKS}")
endfunction()

foreach(_var IN ITEMS "cunumeric_LEGATE_CORE_VERSION"
                      "cunumeric_LEGATE_CORE_BRANCH"
                      "cunumeric_LEGATE_CORE_REPOSITORY"
                      "cunumeric_EXCLUDE_LEGATE_CORE_FROM_ALL")
  if(DEFINED ${_var})
    # Create a cunumeric_LEGATE_CORE_BRANCH variable in the current scope either from the existing
    # current-scope variable, or the cache variable.
    set(${_var} "${${_var}}")
    # Remove cunumeric_LEGATE_CORE_BRANCH from the CMakeCache.txt. This ensures reconfiguring the same
    # build dir without passing `-Dcunumeric_LEGATE_CORE_BRANCH=` reverts to the value in versions.json
    # instead of reusing the previous `-Dcunumeric_LEGATE_CORE_BRANCH=` value.
    unset(${_var} CACHE)
  endif()
endforeach()

if(NOT DEFINED cunumeric_LEGATE_CORE_VERSION)
  set(cunumeric_LEGATE_CORE_VERSION "${cunumeric_VERSION}")
endif()

find_or_configure_legate_core(VERSION          ${cunumeric_LEGATE_CORE_VERSION}
                              REPOSITORY       ${cunumeric_LEGATE_CORE_REPOSITORY}
                              BRANCH           ${cunumeric_LEGATE_CORE_BRANCH}
                              EXCLUDE_FROM_ALL ${cunumeric_EXCLUDE_LEGATE_CORE_FROM_ALL}
)
