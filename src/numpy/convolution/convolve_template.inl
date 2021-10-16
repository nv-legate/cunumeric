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

#include <map>
#include "numpy/pitches.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <VariantKind KIND, LegateTypeCode CODE, int DIM>
struct ConvolveImplBody;

template <VariantKind KIND>
struct ConvolveImpl {
  template <LegateTypeCode CODE, int DIM, std::enable_if_t<(DIM <= 3)>* = nullptr>
  void operator()(ConvolveArgs& args) const
  {
    using VAL        = legate_type_of<CODE>;
    auto subrect     = args.out.shape<DIM>();
    auto filter_rect = args.filter.shape<DIM>();

    if (subrect.empty()) return;

    auto input_subrect = subrect;
    for (auto idx = 1; idx < args.inputs.size(); ++idx) {
      auto image_subrect = args.inputs[idx].shape<DIM>();
      input_subrect      = input_subrect.union_bbox(image_subrect);
    }

    auto out    = args.out.write_accessor<VAL, DIM>(subrect);
    auto filter = args.filter.read_accessor<VAL, DIM>(filter_rect);
    // This is valid only because we colocate all shifted images with the main tile
    auto input = args.inputs[0].read_accessor<VAL, DIM>(input_subrect);

    Rect<DIM> root_rect(args.root_domain);
    ConvolveImplBody<KIND, CODE, DIM>()(out, filter, input, root_rect, subrect, filter_rect);
  }

  template <LegateTypeCode CODE, int DIM, std::enable_if_t<!(DIM <= 3)>* = nullptr>
  void operator()(ConvolveArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void convolve_template(TaskContext& context)
{
  ConvolveArgs args;

  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();

  args.out    = std::move(outputs[0]);
  args.filter = std::move(inputs[0]);
  for (uint32_t idx = 1; idx < inputs.size(); ++idx) args.inputs.push_back(std::move(inputs[idx]));

  auto shape           = context.scalars()[0].value<DomainPoint>();
  args.root_domain.dim = shape.dim;
  for (int32_t dim = 0; dim < shape.dim; ++dim) {
    args.root_domain.rect_data[dim]             = 0;
    args.root_domain.rect_data[dim + shape.dim] = shape[dim] - 1;
  }

  double_dispatch(args.out.dim(), args.out.code(), ConvolveImpl<KIND>{}, args);
}

template<typename VAL, int DIM>
static void
compute_tiles(Point<DIM>& output_tile,
              Point<DIM>& filter_tile,
              const Point<DIM>& output_bounds,
              const Point<DIM>& filter_bounds,
              const unsigned max_size,
              const unsigned min_lastdim_bytes)
{
  assert(min_lastdim_bytes <= max_size);
  const unsigned max_elements = max_size / sizeof(VAL);
  if (DIM == 1) {
    // Figure out how many elements we can fit in half the max size
    // for the output tile
    output_tile[0] = max_elements/2;
    if (output_bounds[0] < output_tile[0])
      output_tile[0] = output_bounds[0];
    // Check to see if we've covered the filter or not
    if (output_tile[0] < filter_bounds[0]) {
      // Output tile is less than the filter tile, so we're
      // going to need to tile the filter also in the other
      // half of the available space
      filter_tile[0] = max_elements/2;
      if (filter_bounds[0] < filter_tile[0])
        filter_tile[0] = filter_bounds[0];
    } else {
      // Output tile covers the entire filter so we can continue
      // growing the output tile to cover the remaining elements
      filter_tile[0] = filter_bounds[0];
      output_tile[0] = max_elements - filter_tile[0];
      if (output_bounds[0] < output_tile[0])
        output_tile[0] = output_bounds[0];
    }
  } else {
    // Figure out how many output elements we can fit in half the max size
    output_tile[DIM-1] = min_lastdim_bytes / sizeof(VAL);
    if (output_bounds[DIM-1] < output_tile[DIM-1])
      output_tile[DIM-1] = output_bounds[DIM-1];
    for (int d = 0; d < (DIM-1); d++)
      output_tile[d] = 1;
    // Double all but the last dimensions until we get all the dimensions
    // equal to the last dimension or we can't grow bigger
    bool done = false;
    while (!done) {
      done = true;
      for (int d = DIM-2; d >= 0; d--) {
        if (output_tile[d] == output_tile[DIM-1])
          continue;
        if (output_tile[d] == output_bounds[d])
          continue;
        // Try doubling the tile on this dimension
        // and then clamp if necessary
        const unsigned old = output_tile[d];
        output_tile[d] *= 2;
        if (output_bounds[d] < output_tile[d])
          output_tile[d] = output_bounds[d];
        if (output_tile[DIM-1] < output_tile[d])
          output_tile[d] = output_tile[DIM-1];
        unsigned next_size = 1;
        for (int d2 = 0; d2 < DIM; d2++)
          next_size *= output_tile[d2];
        if (next_size <= (max_elements/2))
          done = false; // We changed the tiling so we're not done
        else
          output_tile[d] = old;
      }
    }
    // We'll iterate dimensions based on how big the filter is in
    // that dimension to try to increase our chances of covering
    // the entire filter
    std::multimap<coord_t,int> filterdims;
    for (int d = 0; d < DIM; d++)
      filterdims.insert(std::make_pair(filter_bounds[d], d));
    int index = 0;
    int dimorder[DIM];
    for (std::multimap<coord_t,int>::reverse_iterator it =
          filterdims.rbegin(); it != filterdims.rend(); it++)
      dimorder[index++] = it->second;
    // We've tried to balance out the dimensions, so now try to double
    // all of the dimensions in row-major order until we can't make it
    // the tile any larger
    done = false;
    while (!done) {
      done = true;
      for (int d = 0; d < DIM; d++) {
        int dim = dimorder[d];
        if (output_tile[dim] == output_bounds[dim])
          continue;
        // Try doubling the tile on this dimension
        // and then clamp if necessary
        const unsigned old = output_tile[dim];
        output_tile[dim] *= 2;
        if (output_bounds[dim] < output_tile[dim])
          output_tile[dim] = output_bounds[dim];
        unsigned next_size = 1;
        for (int d2 = 0; d2 < DIM; d2++)
          next_size *= output_tile[d2];
        if (next_size <= (max_elements/2))
          done = false; // We changed the tiling so we're not done
        else
          output_tile[dim] = old;
      }
    }
    // Finally try to grow dimensions incrementally up to the bounds
    done = false;
    while (!done) {
      done = true;
      for (int d = 0; d < DIM; d++) {
        int dim = dimorder[d];
        if (output_tile[dim] == output_bounds[dim])
          continue;
        output_tile[dim]++;
        unsigned next_size = 1;
        for (int d2 = 0; d2 < DIM; d2++)
          next_size *= output_tile[d2];
        if (next_size <= (max_elements/2))
          done = false; // We changed the tiling so we're not done
        else
          output_tile[dim]--;
      }
    }
    // Check to see if we've covered the filter or not
    bool filter_covered = true;
    for (int d = 0; d < DIM; d++) {
      if (filter_bounds[d] <= output_tile[d])
        continue;
      filter_covered = false;
      break;
    }
    if (filter_covered) {
      unsigned filter_elements = 1;
      for (int d = 0; d < DIM; d++) {
        filter_tile[d] = filter_bounds[d];
        filter_elements *= filter_bounds[d];
      }
      // Figure out what our new size bounds is since we covered the filter
      const unsigned out_elements = max_elements - filter_elements; 
      // Go back to prefering row major dimensions here
      // Start with powers of two increases again
      done = false;
      while (!done) {
        done = true;
        for (int d = DIM-1; d >= 0; d--) {
          if (output_tile[d] == output_bounds[d])
            continue;
          // Try doubling the tile on this dimension
          // and then clamp if necessary
          const unsigned old = output_tile[d];
          output_tile[d] *= 2;
          if (output_bounds[d] < output_tile[d])
            output_tile[d] = output_bounds[d];
          unsigned next_size = 1;
          for (int d2 = 0; d2 < DIM; d2++)
            next_size *= output_tile[d2];
          if (next_size <= out_elements)
            done = false; // We changed the tiling so we're not done
          else
            output_tile[d] = old;
        }
      }
      // Now do incremental updates
      done = false;
      while (!done) {
        done = true;
        for (int d = DIM-1; d >= 0; d--) {
          if (output_tile[d] == output_bounds[d])
            continue;
          output_tile[d]++;
          unsigned next_size = 1;
          for (int d2 = 0; d2 < DIM; d2++)
            next_size *= output_tile[d2];
          if (next_size <= out_elements)
            done = false; // We changed the tiling so we're not done
          else
            output_tile[d]--;
        }
      }
    } else {
      // We didn't cover the filter
      // Start by tiling the filter the same as the output
      // but clamp any bounds
      for (int d = 0; d < DIM; d++) {
        filter_tile[d] = output_tile[d];
        if (filter_bounds[d] < filter_tile[d])
          filter_tile[d] = filter_bounds[d];
      }
      // Try to double any remaining dimensions
      done = false;
      while (!done) {
        done = true;
        for (int d = DIM-1; d >= 0; d--) {
          if (filter_tile[d] == filter_bounds[d])
            continue;
          // Try doubling the tile on this dimension
          // and then clamp if necessary
          const unsigned old = filter_tile[d];
          filter_tile[d] *= 2;
          if (filter_bounds[d] < filter_tile[d])
            filter_tile[d] = filter_bounds[d];
          unsigned next_size = 1;
          for (int d2 = 0; d2 < DIM; d2++)
            next_size *= filter_tile[d2];
          if (next_size <= (max_elements/2))
            done = false; // We changed the tiling so we're not done
          else
            filter_tile[d] = old;
        }
      }
      // Finally try to grow dimensions incrementally up to the bounds
      done = false;
      while (!done) {
        done = true;
        for (int d = DIM-1; d >= 0; d--) {
          if (filter_tile[d] == filter_bounds[d])
            continue;
          filter_tile[d]++;
          unsigned next_size = 1;
          for (int d2 = 0; d2 < DIM; d2++)
            next_size *= filter_tile[d2];
          if (next_size <= (max_elements/2))
            done = false; // We changed the tiling so we're not done
          else
            filter_tile[d]--;
        }
      }
    }
  }
}

template<typename VAL, int DIM>
static unsigned 
compute_filter_tile(Point<DIM>& tile,
                    const Point<DIM>& bounds,
                    const Point<DIM>& output,
                    const size_t max_size)
{
  // Try doubling dimensions until we can't make it any larger
  unsigned result = 0;
  bool done = false;
  while (!done) {
    done = true;
    for (int d = DIM-1; d >= 0; d--) {
      // Skip if it will exceed our bounds
      if (bounds[d] < (2*tile[d]))
        continue;
      unsigned filter_size = sizeof(VAL); 
      tile[d] *= 2;
      for (int d2 = 0; d2 < DIM; d2++)
        filter_size *= tile[d2];
      unsigned input_size = sizeof(VAL);
      for (int d2 = 0; d2 < DIM; d2++)
        input_size *= (output[d2] + 2*(tile[d2]/2));
      unsigned total_size = filter_size + input_size;
      if (total_size <= max_size) {
        result = total_size;
        done = false;
      } else
        tile[d] /= 2;
    }
  }
  // Then try incrementally increasing dimensions until we can't 
  // make it any larger
  done = false;
  while (!done) {
    done = true;
    for (int d = DIM-1; d >= 0; d--) {
      // Skip if it will exceed our bounds
      if (bounds[d] == tile[d])
        continue;
      unsigned filter_size = sizeof(VAL);
      tile[d]++;
      for (int d2 = 0; d2 < DIM; d2++)
        filter_size *= tile[d2];
      unsigned input_size = sizeof(VAL);
      for (int d2 = 0; d2 < DIM; d2++)
        input_size *= (output[d2] + 2*(tile[d2]/2));
      unsigned total_size = filter_size + input_size;
      if (total_size <= max_size) {
        result = total_size;
        done = false;
      } else
        tile[d]--;
    }
  }
  return result;
}

template<typename VAL, int DIM>
static unsigned 
roundup_tile(Point<DIM>& tile,
             const Point<DIM>& bounds,
             const Point<DIM>& padding,
             const unsigned max_size)
{
  if (DIM == 1) {
    // In this single case we can just solve for this directly
    unsigned elements = max_size / sizeof(VAL);
    assert(elements > padding[0]);
    if (tile[0] < (elements - padding[0])) {
      tile[0] = elements - padding[0];
      if (bounds[0] < tile[0])
        tile[0] = bounds[0];
    }
    return (tile[0] + padding[0]) * sizeof(VAL);
  } else {
    // Find the two smallest dimensions and increase one of them
    // until we hit the second smallest one or exceed max_smem_size
    unsigned result = 0;
    unsigned skipdims = 0;
    bool all_same = true;
    while (true) {
      int d1 = DIM-1, d2 = -1;
      int t1 = tile[d1], t2 = 0;
      for (int d = DIM-2; d >= 0; d--) {
        if (skipdims & (1 << d))
          continue;
        // Skip any dimension that is at its bound
        if (tile[d] == bounds[d]) {
          skipdims |= (1 << d);
          continue;
        }
        if (tile[d] < t1) {
          d2 = d1;
          t2 = t1;
          d1 = d;
          t1 = tile[d];
        } else if ((d2 < 0) || (tile[d] < t2)) {
          d2 = d;
          t2 = tile[d];
        }
      }
      if (d2 == -1) {
        // All the other dimensions are at their bounds, check that
        // the last dimension is also at its bound if not solve
        unsigned pitch = sizeof(VAL);
        for (int d = 0; d < (DIM-1); d++)
          pitch *= (tile[d] + padding[d]); 
        // Make sure the last dimension is as large as it can go too
        if (tile[DIM-1] < bounds[DIM-1]) {
          unsigned elements = max_size / pitch;
          assert(elements > padding[DIM-1]);
          assert(tile[DIM-1] < (elements - padding[DIM-1]));
          tile[DIM-1] = elements - padding[DIM-1];
          if (bounds[DIM-1] < tile[DIM-1])
            tile[DIM-1] = bounds[DIM-1];
        }
        return pitch * (tile[DIM-1] + padding[DIM-1]);
      }
      // If we ever get two dimensions of the same size then we know
      // that there is no smallest dimension so we can march all the
      // dimensions together at this point
      if (t1 == t2)
        break;
      // Solve for the max we can walk 
      unsigned pitch = sizeof(VAL);
      for (int d = 0; d < DIM; d++)
        if (d != d1)
          pitch *= (tile[d] + padding[d]);
      unsigned elements = max_size / pitch;
      if ((elements <= padding[d1]) ||
          (t1 >= (elements - padding[d1]))) {
        skipdims |= (1 << d1);
        continue;
      }
      unsigned bound = elements - padding[d1];
      if (bounds[d1] < bound) {
        tile[d1] = bounds[d1];
        result = pitch * (tile[d1] + padding[d1]);
      } else if (bound < t2) {
        tile[d1] = bound;
        result = pitch * (bound + padding[d1]);
        all_same = false;
        break;
      } else {
        tile[d1] = t2;
        result = pitch * (t2 + padding[d1]);
      }
    }
    if (all_same) {
      // Step all the dimensions together until we hit
      // the shared memory upper bound we're targetting
      // This algorithm is in theory slow, but the max
      // memory sizes of caches are "small" and the amount
      // of memory will grow polynomially in the number
      // of dimensions so it should converge quickly
      while (true) {
        unsigned next_size = sizeof(VAL);
        for (int d = 0; d < DIM; d++)
          if (skipdims & (1 << d))
            next_size *= (tile[d] + padding[d]);
          else if ((tile[d] + 1) == bounds[d]) {
            next_size *= (tile[d] + padding[d]);
            skipdims |= (1 << d);
          } else
            next_size *= (tile[d] + 1 + padding[d]);
        if ((next_size > max_size) || (next_size == result)) 
          break;
        result = next_size;
        for (int d = 0; d < DIM; d++)
          tile[d]++;
      }
    }
    return result;
  }
}

}  // namespace numpy
}  // namespace legate
