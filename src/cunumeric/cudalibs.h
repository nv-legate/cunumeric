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

#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cutensor.h>

namespace cunumeric {

struct CUDALibraries {
 public:
  CUDALibraries();
  ~CUDALibraries();

 private:
  // Prevent copying and overwriting
  CUDALibraries(const CUDALibraries& rhs) = delete;
  CUDALibraries& operator=(const CUDALibraries& rhs) = delete;

 public:
  void finalize();
  cudaStream_t get_cached_stream();
  cublasHandle_t get_cublas();
  cusolverDnHandle_t get_cusolver();
  cutensorHandle_t* get_cutensor();

 private:
  void finalize_cublas();
  void finalize_cusolver();
  void finalize_cutensor();

 private:
  bool finalized_;
  cudaStream_t stream_;
  cublasContext* cublas_;
  cusolverDnContext* cusolver_;
  cutensorHandle_t* cutensor_;
};

}  // namespace cunumeric
