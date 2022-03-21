/* Copyright 2021-2022 NVIDIA Corporation
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

#pragma once

#include <curand.h>

#define CHECK_CURAND(expr)                      \
  do {                                          \
    curandStatus_t __result__ = (expr);            \
    check_curand(__result__, __FILE__, __LINE__); \
  } while (false)

__host__ inline void check_curand(curandStatus_t error, const char* file, int line)
{
  if (error != CURAND_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal CURAND failure with error %d in file %s at line %d\n",
            (int)error,
            file,
            line);
    exit((int)error);
  }
}
