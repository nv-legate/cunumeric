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
#include "cunumeric/pitches.h"
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
    Rect<DIM> rect;

  JITKernelStorage& jit_storage =JITKernelStorage::get_instance(); 

  //std::hash<std::string> hasher;
  CUfunction func;
  std::pair<int64_t,DomainPoint> key(args.hash, args.point);
  //size_t ptx_hash = hasher(args.ptx);
  //std::cout <<"IRINA DEBUG within cuda task hash = "<<args.hash<< " , registered = ?"<<jit_storage.registered_jit_funtion(key)<<std::endl;
  if (jit_storage.registered_jit_funtion(key)){
    func = jit_storage.return_saved_jit_function(key);
  }
  else{
    assert(false); //should never come here
   }

    // Filling up the buffer with arguments
    size_t buffer_size = (args.inputs.size()+args.scalars.size()) * sizeof(void*);
    buffer_size +=sizeof(size_t);//size
    buffer_size += sizeof(size_t);//dim
    buffer_size += sizeof(void*);//pitches
    buffer_size += sizeof(void*);//lo_point
    buffer_size += sizeof(void*);//strides

    std::vector<char> arg_buffer(buffer_size);
    char* raw_arg_buffer = arg_buffer.data();

    auto p = raw_arg_buffer;
    size_t strides[DIM];
    size_t size =1;
    if (args.inputs.size()>0){
      rect = args.inputs[0].shape<DIM>();
      size = rect.volume();
      for (size_t i = 0; i < args.inputs.size(); i++) {
        if (i < args.num_outputs) {
          auto out                           = args.outputs[i].write_accessor<VAL, DIM>(rect);
          *reinterpret_cast<const void**>(p) = out.ptr(rect, strides);
        } else {
          auto in                            = args.inputs[i].read_accessor<VAL, DIM>(rect);
          *reinterpret_cast<const void**>(p) = in.ptr(rect, strides);
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
    size_t dim=DIM;
    p += sizeof(size_t);
    memcpy(p, &dim, sizeof(size_t));
    p += sizeof(size_t);
    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);
    //create buffers for pitches, lower point and strides since
    //we need to pass pointer to device memory
    auto device_pitches   = create_buffer<int64_t>(Point<1>(DIM-1), Memory::Kind::Z_COPY_MEM);
    auto device_lo   = create_buffer<int64_t>(Point<1>(DIM), Memory::Kind::Z_COPY_MEM);
    auto device_strides   = create_buffer<int64_t>(Point<1>(DIM), Memory::Kind::Z_COPY_MEM);
    //std::cout<<"IRINA DEBUG"<<std::endl;
    for (size_t i=0; i<DIM;i++){
      if (i!=DIM-1){
        device_pitches[Point<1>(i)]=pitches.data()[i];
        //std::cout<<" pitches ="<<pitches.data()[i];
        }
      device_lo[Point<1>(i)]=rect.lo[i];
      device_strides[Point<1>(i)] = strides[i];
      //std::cout<<" device_lo = " <<rect.lo[i]<< "  strides = "<<strides[i]<<std::endl;
    }
    *reinterpret_cast<const void**>(p) =device_pitches.ptr(Point<1>(0));
    p += sizeof(void*);
    *reinterpret_cast<const void**>(p) =device_lo.ptr(Point<1>(0));
    p += sizeof(void*);
    *reinterpret_cast<const void**>(p) =device_strides.ptr(Point<1>(0));
    p += sizeof(void*);
    

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
 
  uint32_t num_outputs = context.scalars()[0].value<uint32_t>();
  uint32_t num_scalars = context.scalars()[1].value<uint32_t>();
  std::vector<Scalar>scalars;
  for (size_t i=2; i<(2+num_scalars); i++)
      scalars.push_back(context.scalars()[i]);
  
  int64_t ptx_hash = context.scalars()[2+num_scalars].value<int64_t>();
 // bool is_created = context.scalars()[3+num_scalars].value<bool>();


  EvalUdfArgs args{0,
                   context.inputs(),
                   context.outputs(),
                   scalars,
                   num_outputs,
                   context.get_task_index(),
                   ptx_hash};
  //if (!is_created)
  //    args.ptx = context.scalars()[4+num_scalars].value<std::string>();
  size_t dim=1;
  if (args.inputs.size()>0){
    dim = args.inputs[0].dim() == 0 ? 1 : args.inputs[0].dim();
    double_dispatch(dim, args.inputs[0].code(), EvalUdfGPU{}, args);
  }
  else{
    //FIXME
    double_dispatch(dim, args.inputs[0].code(), EvalUdfGPU{}, args);
    //double_dispatch(dim, 0 , EvalUdfGPU{}, args);
  }
}
}  // namespace cunumeric
