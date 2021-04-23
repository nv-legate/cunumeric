/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "legate.h"
#ifdef LEGATE_USE_CUDA
#include "cuda_libs.h"
#endif
#include "realm/python/python_module.h"
#include <cstdlib>
#include <cstring>

using namespace Legion;

int main(int argc, char** argv)
{
  Runtime::initialize(&argc, &argv);
  Legate::parse_config();
  const char* python_modules_path = getenv("PYTHON_MODULES_PATH");
  // We better have one of these from the driver
  assert(python_modules_path != NULL);
  // do this before any threads are spawned
  char* previous_python_path = getenv("PYTHONPATH");
  if (previous_python_path != 0) {
    size_t bufsize = 8192;
    char* buffer   = (char*)calloc(bufsize, sizeof(char));
    assert(buffer != 0);

    assert(strlen(previous_python_path) + strlen(python_modules_path) + 2 < bufsize);
    // Concatenate PYTHON_MODULES_PATH to the end of PYTHONPATH.
    bufsize--;
    strncat(buffer, previous_python_path, bufsize);
    bufsize -= strlen(previous_python_path);
    strncat(buffer, ":", bufsize);
    bufsize--;
    strncat(buffer, python_modules_path, bufsize);
    bufsize -= strlen(python_modules_path);
    setenv("PYTHONPATH", buffer, true /*overwrite*/);
  } else {
    setenv("PYTHONPATH", python_modules_path, true /*overwrite*/);
  }

#ifdef LEGATE_USE_CUDA
  {
    // Populate the cuda libraries data structure with an entry for
    // every GPU processor so we don't have to synchronize later
    std::map<Processor, CUDALibraries>& cuda_libraries = CUDALibraries::get_libraries();
    Machine::ProcessorQuery gpu_procs(Machine::get_machine());
    gpu_procs.local_address_space();
    gpu_procs.only_kind(Processor::TOC_PROC);
    for (Machine::ProcessorQuery::iterator it = gpu_procs.begin(); it != gpu_procs.end(); it++) {
      CUDALibraries& lib = cuda_libraries[*it];
      assert(!lib.initialized);
    }
  }
#endif

  Realm::Python::PythonModule::import_python_module("legate.core");
  // Record that we need a callback for Legate
  Runtime::add_registration_callback(Legate::registration_callback);
  // Start the Legion runtime
  return Runtime::start(argc, argv);
}
