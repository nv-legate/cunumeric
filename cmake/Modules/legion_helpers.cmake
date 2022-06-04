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

function(get_legion_and_realm_includes _out_var)

  set(realm_includes "")
  if(TARGET Legion::RealmRuntime)
    get_target_property(realm_includes Legion::RealmRuntime INTERFACE_INCLUDE_DIRECTORIES)
  elseif(TARGET Legion::Realm)
    get_target_property(realm_includes Legion::Realm INTERFACE_INCLUDE_DIRECTORIES)
  endif()

  set(legion_includes "")
  if(TARGET Legion::LegionRuntime)
    get_target_property(legion_includes Legion::LegionRuntime INTERFACE_INCLUDE_DIRECTORIES)
  elseif(TARGET Legion::Legion)
    get_target_property(legion_includes Legion::Legion INTERFACE_INCLUDE_DIRECTORIES)
  endif()

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

  set(${_out_var} ${extra_includes} PARENT_SCOPE)
endfunction()
