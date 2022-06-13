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
  set(oneValueArgs VERSION REPOSITORY PINNED_TAG EXCLUDE_FROM_ALL USE_OPENMP)
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
    if(PKG_USE_OPENMP)
      set(tblis_thread_model "--enable-thread-model=openmp")
    endif()

    # CMake sets `ENV{CC}` to /usr/bin/cc if it's not set. This causes tblis'
    # `./configure` to fail. For now, detect this case and reset ENV{CC/CXX}.
    # Remove this workaround when we can use `cmake_policy(SET CMP0132 NEW)`:
    # https://gitlab.kitware.com/cmake/cmake/-/merge_requests/7108

    set(CC_ORIG "$ENV{CC}")
    set(CXX_ORIG "$ENV{CXX}")
    set(_CC "${CMAKE_C_COMPILER}")
    set(_CXX "${CMAKE_CXX_COMPILER}")

    if(CC_ORIG MATCHES "^.*\/cc$")
      file(REAL_PATH "${_CC}" _CC EXPAND_TILDE)
      file(REAL_PATH "${_CXX}" _CXX EXPAND_TILDE)
      set(ENV{CC} "${_CC}")
      set(ENV{CXX} "${_CXX}")
    endif()

    # Use the caching compiler (if provided) to speed up tblis builds
    if(CMAKE_C_COMPILER_LAUNCHER)
      set(ENV{CC} "${CMAKE_C_COMPILER_LAUNCHER} ${_CC}")
    endif()
    if(CMAKE_CXX_COMPILER_LAUNCHER)
      set(ENV{CXX} "${CMAKE_CXX_COMPILER_LAUNCHER} ${_CXX}")
    endif()

    message(VERBOSE "cunumeric: ENV{CC}=\"$ENV{CC}\"")
    message(VERBOSE "cunumeric: ENV{CXX}=\"$ENV{CXX}\"")

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

    # Reset ENV{CC/CXX}
    set(ENV{CC} "${CC_ORIG}")
    set(ENV{CXX} "${CXX_ORIG}")

    # Build and install tblis to ${tblis_BINARY_DIR}
    execute_process(
      COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL} install
      WORKING_DIRECTORY      "${tblis_SOURCE_DIR}"
      COMMAND_ECHO           STDOUT
      COMMAND_ERROR_IS_FATAL ANY
      ECHO_ERROR_VARIABLE
      ECHO_OUTPUT_VARIABLE)
  endif()

  set(lib_suffix "")
  if(BUILD_SHARED_LIBS)
    add_library(tblis SHARED IMPORTED GLOBAL)
    set(lib_suffix "${CMAKE_SHARED_LIBRARY_SUFFIX}")
  else()
    add_library(tblis STATIC IMPORTED GLOBAL)
    set(lib_suffix "${CMAKE_STATIC_LIBRARY_SUFFIX}")
  endif()

  add_library(tblis::tblis ALIAS tblis)
  target_include_directories(tblis INTERFACE ${tblis_BINARY_DIR}/include)
  set_target_properties(tblis
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               IMPORTED_SONAME                     tblis
               IMPORTED_LOCATION                   "${tblis_BINARY_DIR}/lib/libtblis${lib_suffix}"
               INSTALL_REMOVE_ENVIRONMENT_RPATH    ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON)
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
                        USE_OPENMP       ${Legion_USE_OpenMP}
)
