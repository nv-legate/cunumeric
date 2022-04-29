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

function(find_or_configure_tblis)
  set(oneValueArgs VERSION REPOSITORY PINNED_TAG EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  rapids_cpm_find(tblis ${PKG_VERSION}
      GLOBAL_TARGETS      tblis::tblis
      BUILD_EXPORT_SET    cunumeric-exports
      INSTALL_EXPORT_SET  cunumeric-exports
      CPM_ARGS
        GIT_REPOSITORY    ${PKG_REPOSITORY}
        GIT_TAG           ${PKG_PINNED_TAG}
        EXCLUDE_FROM_ALL  ${PKG_EXCLUDE_FROM_ALL}
  )

  if(tblis_ADDED AND (NOT EXISTS "${tblis_BINARY_DIR}/include"))

    # Configure tblis
    set(tblis_thread_model "--disable-thread-model")
    if(OpenMP_FOUND)
      set(tblis_thread_model "--enable-thread-model=openmp")
    endif()

    # CMake sets `ENV{CC}` to /usr/bin/cc if it's not set. This causes tblis'
    # `./configure` to fail. For now, detect this case and unset ENV{CC/CXX}.
    # Remove this workaround when we can use `cmake_policy(SET CMP0132 NEW)`:
    # https://gitlab.kitware.com/cmake/cmake/-/merge_requests/7108

    set(CC_ORIG $ENV{CC})
    set(CXX_ORIG $ENV{CXX})

    if(CC_ORIG MATCHES "^.*\/cc$")
      unset(ENV{CC})
      unset(ENV{CXX})
    endif()

    message(VERBOSE "cunumeric: ENV{CC}=$ENV{CC}")
    message(VERBOSE "cunumeric: ENV{CXX}=$ENV{CXX}")

    execute_process(
      COMMAND ./configure
        ${tblis_thread_model}
        --enable-silent-rules
        --disable-option-checking
        --with-label-type=int32_t
        --with-length-type=int64_t
        --with-stride-type=int64_t
        --prefix=${tblis_BINARY_DIR}
        --disable-dependency-tracking
      WORKING_DIRECTORY      "${tblis_SOURCE_DIR}"
      COMMAND_ECHO           STDOUT
      COMMAND_ERROR_IS_FATAL ANY
      ECHO_ERROR_VARIABLE
      ECHO_OUTPUT_VARIABLE)

    # Build and install tblis to ${tblis_BINARY_DIR}
    execute_process(
      COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL} install
      WORKING_DIRECTORY      "${tblis_SOURCE_DIR}"
      COMMAND_ECHO           STDOUT
      COMMAND_ERROR_IS_FATAL ANY
      ECHO_ERROR_VARIABLE
      ECHO_OUTPUT_VARIABLE)

    # Reset ENV{CC/CXX}
    if(CC_ORIG)
      set(ENV{CC} ${CC_ORIG})
    endif()
    if(CXX_ORIG)
      set(ENV{CXX} ${CXX_ORIG})
    endif()
  endif()

  add_library(tblis INTERFACE IMPORTED)
  add_library(tblis::tblis ALIAS tblis)
  target_include_directories(tblis INTERFACE ${tblis_BINARY_DIR}/include)
  set_target_properties(tblis PROPERTIES IMPORTED_LOCATION ${tblis_BINARY_DIR}/lib/libtblis.so)
endfunction()

if(NOT DEFINED CUNUMERIC_TBLIS_BRANCH)
  set(CUNUMERIC_TBLIS_BRANCH master)
endif()

if(NOT DEFINED CUNUMERIC_TBLIS_REPOSITORY)
  set(CUNUMERIC_TBLIS_REPOSITORY https://github.com/devinamatthews/tblis.git)
endif()

find_or_configure_tblis(VERSION          1.2.0
                        REPOSITORY       ${CUNUMERIC_TBLIS_REPOSITORY}
                        PINNED_TAG       ${CUNUMERIC_TBLIS_BRANCH}
                        EXCLUDE_FROM_ALL ${CUNUMERIC_EXCLUDE_TBLIS_FROM_ALL}
)
