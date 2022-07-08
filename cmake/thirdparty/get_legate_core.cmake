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

function(find_or_configure_legate_core)
  set(oneValueArgs VERSION REPOSITORY BRANCH EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(FIND_PKG_ARGS      ${PKG_VERSION}
      GLOBAL_TARGETS     legate::core
      BUILD_EXPORT_SET   cunumeric-exports
      INSTALL_EXPORT_SET cunumeric-exports)

  # First try to find legate_core via find_package()
  # so the `Legion_USE_*` variables are visible
  rapids_find_package(legate_core ${FIND_PKG_ARGS} QUIET)

  if(legate_core_FOUND)
    message(STATUS "CPM: using local package legate_core@${PKG_VERSION}")
  else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cpm_helpers.cmake)
    get_cpm_git_args(legate_core_cpm_git_args REPOSITORY ${PKG_REPOSITORY} BRANCH ${PKG_BRANCH})
    rapids_cpm_find(legate_core ${FIND_PKG_ARGS}
        CPM_ARGS
          ${legate_core_cpm_git_args}
          EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
    )
  endif()

  set(Legion_USE_CUDA ${Legion_USE_CUDA} PARENT_SCOPE)
  set(Legion_USE_OpenMP ${Legion_USE_OpenMP} PARENT_SCOPE)
  set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS} PARENT_SCOPE)

  message(VERBOSE "Legion_USE_CUDA=${Legion_USE_CUDA}")
  message(VERBOSE "Legion_USE_OpenMP=${Legion_USE_OpenMP}")
  message(VERBOSE "Legion_BOUNDS_CHECKS=${Legion_BOUNDS_CHECKS}")
endfunction()

if(NOT DEFINED cunumeric_LEGATE_CORE_BRANCH)
  # TODO: Undo this! Only setting temporarily so CI passes
  set(cunumeric_LEGATE_CORE_BRANCH fea/add-cmake)
  # set(cunumeric_LEGATE_CORE_BRANCH branch-22.07)
endif()

if(NOT DEFINED cunumeric_LEGATE_CORE_REPOSITORY)
  # TODO: Undo this! Only setting temporarily so CI passes
  set(cunumeric_LEGATE_CORE_REPOSITORY https://github.com/trxcllnt/legate.core.git)
  # set(cunumeric_LEGATE_CORE_REPOSITORY https://github.com/nv-legate/legate.core.git)
endif()

find_or_configure_legate_core(VERSION          ${cunumeric_VERSION}
                              REPOSITORY       ${cunumeric_LEGATE_CORE_REPOSITORY}
                              BRANCH           ${cunumeric_LEGATE_CORE_BRANCH}
                              EXCLUDE_FROM_ALL ${cunumeric_EXCLUDE_LEGATE_CORE_FROM_ALL}
)
