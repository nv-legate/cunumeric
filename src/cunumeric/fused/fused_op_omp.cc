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

#include "cunumeric/fused/fused_op.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

/*static*/ void FusedOpTask::omp_variant(TaskContext& context){

  int nOps = context.fusionMetadata.nOps;
  auto opIDs = context.fusionMetadata.opIDs;
  auto offsets = context.fusionMetadata.offsets;
  for (int i = 0; i < nOps; i++) {
    std::vector<Legion::PhysicalRegion> regions;
    // create new context
    const Legion::Task* task = (Legion::Task*)context.task_;

    // pack inputs
    std::vector<Store> inputs;
    auto inputStarts     = context.fusionMetadata.inputStarts;
    auto outputStarts    = context.fusionMetadata.outputStarts;
    auto offsetStarts    = context.fusionMetadata.offsetStarts;
    auto reductionStarts = context.fusionMetadata.reductionStarts;
    unsigned nInputs = (inputStarts[i + 1] - inputStarts[i]);  // want to pack this as a 32 bit uint
    for (unsigned j = 0; j < nInputs; j++) {
      int offsetStart = offsetStarts[i];
      int inputStart  = inputStarts[i];
      int bufferID    = offsets[offsetStart + j] - 1;
      Store& input    = context.inputs()[inputStart + bufferID];
      inputs.push_back(std::move(input));
    }

    // pack outputs
    std::vector<Store> outputs;
    unsigned nOutputs =
      (outputStarts[i + 1] - outputStarts[i]);  // want to pack this as a 32 bit uint
    for (unsigned j = 0; j < nOutputs; j++) {
      int offsetStart = offsetStarts[i];
      int outputStart = outputStarts[i];
      int bufferID    = offsets[offsetStart + nInputs + j];
      bufferID        = (-bufferID) - 1;
      Store& output   = context.outputs()[outputStart + bufferID];
      outputs.push_back(std::move(output));
    }

    // pack reductions
    std::vector<Store> reductions;
    int32_t nReductions = (reductionStarts[i + 1] - reductionStarts[i]);
    for (unsigned j = 0; j < nReductions; j++) {
      int offsetStart    = offsetStarts[i];
      int reductionStart = reductionStarts[i];
      int bufferID       = offsets[offsetStart + nInputs + nOutputs + j];
      // all buffer ids are 1 -indexed
      // negative id is an output, while a positive id is an output
      if (bufferID < 0) {
        bufferID         = (-bufferID) - 1;
        Store& reduction = context.reductions()[reductionStart + bufferID];
        reductions.push_back(std::move(reduction));
      }
    }

    // pack scalars
    auto scalarStarts = context.fusionMetadata.scalarStarts;
    int32_t nScalars  = (scalarStarts[i + 1] - scalarStarts[i]);
    std::vector<Scalar> scalars;
    for (unsigned j = 0; j < nScalars; j++) {
      scalars.push_back(std::move(context.scalars()[scalarStarts[i] + j]));
    }

    TaskContext context3(
      task, (const std::vector<Legion::PhysicalRegion>)regions);  // inputs, outputs, scalars);
    context3.inputs_     = std::move(inputs);
    context3.outputs_    = std::move(outputs);
    context3.reductions_ = std::move(reductions);
    context3.scalars_    = std::move(scalars);

    // launch
    auto descp = Core::ompDescriptors.find(opIDs[i]);

    auto desc = descp->second;
    desc(context3);
    for (unsigned j = 0; j < nOutputs; j++) {
      int offsetStart                          = offsetStarts[i];
      int outputStart                          = outputStarts[i];
      int bufferID                             = offsets[offsetStart + nInputs + j];
      bufferID                                 = (-bufferID) - 1;
      context.outputs_[outputStart + bufferID] = std::move(context3.outputs_[j]);
    }
  }
}

}  // namespace cunumeric
