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

##############################################################################
# - User Options  ------------------------------------------------------------

option(FIND_CUNUMERIC_CPP "Search for existing cuNumeric C++ installations before defaulting to local files"
       OFF)

##############################################################################
# - Dependencies -------------------------------------------------------------

# If the user requested it we attempt to find cunumeric.
if(FIND_CUNUMERIC_CPP)
  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${cunumeric_version} cunumeric parsed_ver)
  rapids_find_package(cunumeric ${parsed_ver} EXACT CONFIG
                      GLOBAL_TARGETS     cunumeric::cunumeric
                      BUILD_EXPORT_SET   cunumeric-python-exports
                      INSTALL_EXPORT_SET cunumeric-python-exports)
else()
  set(cunumeric_FOUND OFF)
endif()

if(NOT cunumeric_FOUND)
  set(SKBUILD OFF)
  set(Legion_USE_Python ON)
  set(Legion_BUILD_BINDINGS ON)
  add_subdirectory(. "${CMAKE_CURRENT_SOURCE_DIR}/build")
  set(SKBUILD ON)
endif()

add_custom_target("generate_install_info_py" ALL
  COMMAND ${CMAKE_COMMAND}
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_install_info_py.cmake"
  COMMENT "Generate install_info.py"
  VERBATIM
)

add_library(cunumeric_python INTERFACE)
add_library(cunumeric::cunumeric_python ALIAS cunumeric_python)
target_link_libraries(cunumeric_python INTERFACE legate::core)

##############################################################################
# - install targets ----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS cunumeric_python
        DESTINATION ${lib_dir}
        EXPORT cunumeric-python-exports)

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide Python targets for cuNumeric, an aspiring drop-in replacement for NumPy at scale.

Imported Targets:
  - cunumeric::cunumeric_python

]=])

set(code_string "")

rapids_export(
  INSTALL cunumeric_python
  EXPORT_SET cunumeric-python-exports
  GLOBAL_TARGETS cunumeric_python
  NAMESPACE cunumeric::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD cunumeric_python
  EXPORT_SET cunumeric-python-exports
  GLOBAL_TARGETS cunumeric_python
  NAMESPACE cunumeric::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)
