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

#include "cunumeric/sort/sort.h"
#include "cunumeric/sort/sort_template.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

// general routine
template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(VAL* inptr,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  bool is_index_space,
                  Legion::DomainPoint index_point,
                  Legion::Domain domain)
  {
    // std::cout << "local size = " << volume << ", dist. = " << is_index_space << ", index_point =
    // "
    //           << index_point << ", domain/volume = " << domain << "/" << domain.get_volume() <<
    //           std::endl;

    std::sort(inptr, inptr + volume);

    // in case of distributed data we need to switch to sample sort
    if (is_index_space) {
      // create (starting) sample of (at most) domain.get_volume() equidistant values
      // also enrich values with additional indexes rank & local position in order to handle
      // duplicate values
      size_t num_local_samples = std::min(domain.get_volume(), volume);
      size_t local_rank        = index_point[0];
      auto local_samples       = std::make_unique<SampleEntry<VAL>[]>(num_local_samples);
      for (int i = 0; i < num_local_samples; ++i) {
        const size_t index        = (i + 1) * volume / num_local_samples - 1;
        local_samples[i].value    = inptr[index];
        local_samples[i].rank     = local_rank;
        local_samples[i].local_id = index;
      }

      // std::cout << "local samples: size = " << num_local_samples << std::endl;
      // std::cout << "first = (" << local_samples[0].value << "," << local_samples[0].rank << ","<<
      // local_samples[0].local_id << ")" << std::endl; std::cout << "last = (" <<
      // local_samples[num_local_samples-1].value << "," << local_samples[num_local_samples-1].rank
      // << ","<< local_samples[num_local_samples-1].local_id << ")" << std::endl;

      // all2all those samples
      // TODO broadcast package size
      // TODO allocate targets
      // TODO broadcast samples
      size_t num_global_samples = 15;
      std::unique_ptr<SampleEntry<VAL>[]> global_samples(new SampleEntry<VAL>[num_global_samples]);

      // sort all samples (utilize 2nd and 3rd sort criteria as well)
      std::sort(&(global_samples[0]),
                &(global_samples[0]) + num_global_samples,
                SampleEntryComparator<VAL>());

      // define splitters
      auto splitters = std::make_unique<SampleEntry<VAL>[]>(domain.get_volume() - 1);
      for (int i = 0; i < domain.get_volume() - 1; ++i) {
        const size_t index = (i + 1) * num_global_samples / domain.get_volume() - 1;
        splitters[i]       = global_samples[index];
      }

      do {
        // compute local package sizes for every process based on splitters
        std::unique_ptr<size_t[]> local_partition_size(new size_t[domain.get_volume()]);
        {
          size_t range_start    = 0;
          size_t local_position = 0;
          for (int p_index = 0; p_index < domain.get_volume(); ++p_index) {
            // move as long current value is lesser or equaöl to current splitter
            while (local_position < volume &&
                   (inptr[local_position] < splitters[p_index].value ||
                    (inptr[local_position] == splitters[p_index].value &&
                     (local_rank < splitters[p_index].rank ||
                      (local_rank == splitters[p_index].rank &&
                       local_position <= splitters[p_index].local_id))))) {
              local_position++;
            }

            local_partition_size[p_index++] = local_position - range_start;
            range_start                     = local_position;
          }
        }

        // communicate local package-sizes all2all
        // TODO

        // evaluate distribution result??
        // TODO

        // if (good enough) break;
        // TODO
        break;
        // else iterate/improve splitters
        // TODO

      } while (true);

      // all2all accepted distribution
      // package sizes should already be known
      // all2all communication
      // TODO

      // final merge sort of received packages
      // TODO
    }
  }
};

/*static*/ void SortTask::cpu_variant(TaskContext& context)
{
  sort_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { SortTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
