/* Copyright 2023 NVIDIA Corporation
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

#include "cunumeric/vectorize/eval_udf.h"
#include "cunumeric/cuda_help.h"
#include <regex>
#include <cuda.h>

namespace cunumeric {

using namespace Legion;
using namespace legate;

struct EvalUdfGPU {
  template <LegateTypeCode CODE, int DIM>
  void operator()(EvalUdfArgs& args) const
  {
    using VAL = legate_type_of<CODE>;
    //auto rect = args.inputs[0].shape<DIM>();
    //if (rect.empty()) return;

    // 1: we need to vreate a function from the ptx generated y numba
    const unsigned num_options   = 4;
    const size_t log_buffer_size = 16384;
    std::vector<char> log_info_buffer(log_buffer_size);
    std::vector<char> log_error_buffer(log_buffer_size);
    CUjit_option jit_options[] = {
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    };
    void* option_vals[] = {
      static_cast<void*>(log_info_buffer.data()),
      reinterpret_cast<void*>(log_buffer_size),
      static_cast<void*>(log_error_buffer.data()),
      reinterpret_cast<void*>(log_buffer_size),
    };

    CUmodule module;
    CUresult result =
      cuModuleLoadDataEx(&module, args.ptx.data(), num_options, jit_options, option_vals);
    if (result != CUDA_SUCCESS) {
      if (result == CUDA_ERROR_OPERATING_SYSTEM) {
        fprintf(stderr,
                "ERROR: Device side asserts are not supported by the "
                "CUDA driver for MAC OSX, see NVBugs 1628896.\n");
        exit(-1);
      } else if (result == CUDA_ERROR_NO_BINARY_FOR_GPU) {
        fprintf(stderr, "ERROR: The binary was compiled for the wrong GPU architecture.\n");
        exit(-1);
      } else {
        fprintf(stderr, "Failed to load CUDA module! Error log: %s\n", log_error_buffer.data());
#if CUDA_VERSION >= 6050
        const char *name, *str;
        assert(cuGetErrorName(result, &name) == CUDA_SUCCESS);
        assert(cuGetErrorString(result, &str) == CUDA_SUCCESS);
        fprintf(stderr, "CU: cuModuleLoadDataEx = %d (%s): %s\n", result, name, str);
#else
        fprintf(stderr, "CU: cuModuleLoadDataEx = %d\n", result);
#endif
        exit(-1);
      }
    }

    std::cmatch line_match;
    bool match =
      std::regex_search(args.ptx.data(), line_match, std::regex(".visible .entry [_a-zA-Z0-9$]+"));
#ifdef DEBUG_CUNUMERIC
    assert(match);
#endif
    const auto& matched_line = line_match.begin()->str();
    auto fun_name = matched_line.substr(matched_line.rfind(" ") + 1, matched_line.size());

    CUfunction func;
    result = cuModuleGetFunction(&func, module, fun_name.c_str());
#ifdef DEBUG_CUNUMERIC
    assert(result == CUDA_SUCCESS);
#endif

    // 2: after fucntion is generated, we can execute it:

    // Filling up the buffer with arguments
    size_t buffer_size = (args.inputs.size()+args.scalars.size()) * sizeof(void*);
    buffer_size += sizeof(size_t);

    std::vector<char> arg_buffer(buffer_size);
    char* raw_arg_buffer = arg_buffer.data();

    auto p = raw_arg_buffer;

    size_t size =1;
    if (args.inputs.size()>0){
      auto rect = args.inputs[0].shape<DIM>();
      size = rect.volume();
      for (size_t i = 0; i < args.inputs.size(); i++) {
        if (i < args.num_outputs) {
          auto out                           = args.outputs[i].write_accessor<VAL, DIM>(rect);
          *reinterpret_cast<const void**>(p) = out.ptr(rect);
        } else {
          auto in                            = args.inputs[i].read_accessor<VAL, DIM>(rect);
          *reinterpret_cast<const void**>(p) = in.ptr(rect);
        }
        p += sizeof(void*);
      }
    }
    for (auto scalar: args.scalars){
        memcpy(p, scalar.ptr(), scalar.size());
        p += scalar.size();
       // *reinterpret_cast<const void**>(p) =s;
        //p += sizeof(void*);
      }
    memcpy(p, &size, sizeof(size_t));

    void* config[] = {
      CU_LAUNCH_PARAM_BUFFER_POINTER,
      static_cast<void*>(raw_arg_buffer),
      CU_LAUNCH_PARAM_BUFFER_SIZE,
      &buffer_size,
      CU_LAUNCH_PARAM_END,
    };

    const uint32_t gridDimX = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const uint32_t gridDimY = 1;
    const uint32_t gridDimZ = 1;

    const uint32_t blockDimX = THREADS_PER_BLOCK;
    const uint32_t blockDimY = 1;
    const uint32_t blockDimZ = 1;

    auto stream = get_cached_stream();

    // executing the function
    CUresult status = cuLaunchKernel(
      func, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, stream, NULL, config);
    if (status != CUDA_SUCCESS) {
      fprintf(stderr, "Failed to launch a CUDA kernel\n");
      exit(-1);
    }

    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void EvalUdfTask::gpu_variant(TaskContext& context)
{
  std::vector<Scalar>scalars;
  for (size_t i=2; i<context.scalars().size(); i++)
      scalars.push_back(context.scalars()[i]);

  EvalUdfArgs args{0,
                   context.inputs(),
                   context.outputs(),
                   scalars,
                   context.scalars()[0].value<std::string>(),
                   context.scalars()[1].value<uint32_t>()};
  size_t dim=1;
  if (args.inputs.size()>0){
    dim = args.inputs[0].dim() == 0 ? 1 : args.inputs[0].dim();
    double_dispatch(dim, args.inputs[0].code(), EvalUdfGPU{}, args);
  }
  else{
    double_dispatch(dim, args.inputs[0].code(), EvalUdfGPU{}, args);
    //double_dispatch(dim, 0 , EvalUdfGPU{}, args);
  }
}
}  // namespace cunumeric
