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
  set(oneValueArgs VERSION REPOSITORY PINNED_TAG EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(FIND_PKG_ARGS      ${PKG_VERSION}
      GLOBAL_TARGETS     legate::core
      BUILD_EXPORT_SET   cunumeric-exports
      INSTALL_EXPORT_SET cunumeric-exports)

  # First try to find Legion via find_package()
  # so the `Legion_USE_*` variables are visible
  rapids_find_package(legate_core ${FIND_PKG_ARGS} QUIET)

  if(legate_core_FOUND)
    message(STATUS "CPM: using local package legate_core@${PKG_VERSION}")
  else()
    rapids_cpm_find(legate_core ${FIND_PKG_ARGS}
        CPM_ARGS
          GIT_REPOSITORY        ${PKG_REPOSITORY}
          GIT_TAG               ${PKG_PINNED_TAG}
          EXCLUDE_FROM_ALL      ${PKG_EXCLUDE_FROM_ALL}
    )
  endif()

  set(Legion_USE_CUDA ${Legion_USE_CUDA} PARENT_SCOPE)
  set(Legion_USE_OpenMP ${Legion_USE_OpenMP} PARENT_SCOPE)
  set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS} PARENT_SCOPE)

  message(VERBOSE "Legion_USE_CUDA=${Legion_USE_CUDA}")
  message(VERBOSE "Legion_USE_OpenMP=${Legion_USE_OpenMP}")
  message(VERBOSE "Legion_BOUNDS_CHECKS=${Legion_BOUNDS_CHECKS}")
endfunction()

if(NOT DEFINED CUNUMERIC_LEGATE_CORE_BRANCH)
  set(CUNUMERIC_LEGATE_CORE_BRANCH branch-22.07)
endif()

if(NOT DEFINED CUNUMERIC_LEGATE_CORE_REPOSITORY)
  set(CUNUMERIC_LEGATE_CORE_REPOSITORY https://github.com/nv-legate/legate.core)
endif()

find_or_configure_legate_core(VERSION          ${CUNUMERIC_VERSION}
                              REPOSITORY       ${CUNUMERIC_LEGATE_CORE_REPOSITORY}
                              PINNED_TAG       ${CUNUMERIC_LEGATE_CORE_BRANCH}
                              EXCLUDE_FROM_ALL ${CUNUMERIC_EXCLUDE_LEGATE_CORE_FROM_ALL}
)
