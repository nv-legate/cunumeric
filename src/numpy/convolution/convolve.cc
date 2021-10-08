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

#include "numpy/convolution/convolve.h"
#include "numpy/convolution/convolve_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <LegateTypeCode CODE>
struct ConvolveImplBody<VariantKind::CPU, CODE, 1> {
  using VAL = legate_type_of<CODE>;

  void operator()(AccessorWO<VAL, 1> out,
                  AccessorRO<VAL, 1> filter,
                  AccessorRO<VAL, 1> in,
                  const Rect<1>& root_rect,
                  const Rect<1>& subrect,
                  const Rect<1>& filter_rect) const
  {
    assert(filter_rect.lo[0] == 0);

    auto f_extent_x = filter_rect.hi[0] + 1;

    auto center_x = static_cast<coord_t>(f_extent_x / 2);

    auto lo_x = subrect.lo[0];
    auto hi_x = subrect.hi[0];

    auto root_hi_x = root_rect.hi[0];

    for (int64_t out_x = lo_x; out_x <= hi_x; ++out_x) {
      VAL acc{0};
      for (int64_t f_x = 0; f_x < f_extent_x; ++f_x) {
        auto in_x = out_x + f_x - center_x;
        if (in_x < 0 || in_x > root_hi_x) continue;
        acc += in[in_x] * filter[f_extent_x - f_x - 1];
      }
      out[out_x] = acc;
    }
  }
};

template <LegateTypeCode CODE>
struct ConvolveImplBody<VariantKind::CPU, CODE, 2> {
  using VAL = legate_type_of<CODE>;

  void operator()(AccessorWO<VAL, 2> out,
                  AccessorRO<VAL, 2> filter,
                  AccessorRO<VAL, 2> in,
                  const Rect<2>& root_rect,
                  const Rect<2>& subrect,
                  const Rect<2>& filter_rect) const
  {
    assert(filter_rect.lo[0] == 0);
    assert(filter_rect.lo[1] == 0);

    auto f_extent_x = filter_rect.hi[0] + 1;
    auto f_extent_y = filter_rect.hi[1] + 1;

    auto center_x = static_cast<coord_t>(f_extent_x / 2);
    auto center_y = static_cast<coord_t>(f_extent_y / 2);

    auto lo_x = subrect.lo[0];
    auto lo_y = subrect.lo[1];
    auto hi_x = subrect.hi[0];
    auto hi_y = subrect.hi[1];

    auto root_hi_x = root_rect.hi[0];
    auto root_hi_y = root_rect.hi[1];

    for (int64_t out_x = lo_x; out_x <= hi_x; ++out_x)
      for (int64_t out_y = lo_y; out_y <= hi_y; ++out_y) {
        VAL acc{0};
        for (int64_t f_x = 0; f_x < f_extent_x; ++f_x) {
          auto in_x = out_x + f_x - center_x;
          if (in_x < 0 || in_x > root_hi_x) continue;

          for (int64_t f_y = 0; f_y < f_extent_y; ++f_y) {
            auto in_y = out_y + f_y - center_y;
            if (in_y < 0 || in_y > root_hi_y) continue;

            acc += in[in_x][in_y] * filter[f_extent_x - f_x - 1][f_extent_y - f_y - 1];
          }
        }
        out[out_x][out_y] = acc;
      }
  }
};

template <LegateTypeCode CODE>
struct ConvolveImplBody<VariantKind::CPU, CODE, 3> {
  using VAL = legate_type_of<CODE>;

  void operator()(AccessorWO<VAL, 3> out,
                  AccessorRO<VAL, 3> filter,
                  AccessorRO<VAL, 3> in,
                  const Rect<3>& root_rect,
                  const Rect<3>& subrect,
                  const Rect<3>& filter_rect) const
  {
    assert(filter_rect.lo[0] == 0);
    assert(filter_rect.lo[1] == 0);
    assert(filter_rect.lo[2] == 0);

    auto f_extent_x = filter_rect.hi[0] + 1;
    auto f_extent_y = filter_rect.hi[1] + 1;
    auto f_extent_z = filter_rect.hi[2] + 1;

    auto center_x = static_cast<coord_t>(f_extent_x / 2);
    auto center_y = static_cast<coord_t>(f_extent_y / 2);
    auto center_z = static_cast<coord_t>(f_extent_z / 2);

    auto lo_x = subrect.lo[0];
    auto lo_y = subrect.lo[1];
    auto lo_z = subrect.lo[2];
    auto hi_x = subrect.hi[0];
    auto hi_y = subrect.hi[1];
    auto hi_z = subrect.hi[2];

    auto root_hi_x = root_rect.hi[0];
    auto root_hi_y = root_rect.hi[1];
    auto root_hi_z = root_rect.hi[2];

    for (int64_t out_x = lo_x; out_x <= hi_x; ++out_x)
      for (int64_t out_y = lo_y; out_y <= hi_y; ++out_y)
        for (int64_t out_z = lo_z; out_z <= hi_z; ++out_z) {
          VAL acc{0};
          for (int64_t f_x = 0; f_x < f_extent_x; ++f_x) {
            auto in_x = out_x + f_x - center_x;
            if (in_x < 0 || in_x > root_hi_x) continue;

            for (int64_t f_y = 0; f_y < f_extent_y; ++f_y) {
              auto in_y = out_y + f_y - center_y;
              if (in_y < 0 || in_y > root_hi_y) continue;

              for (int64_t f_z = 0; f_z < f_extent_z; ++f_z) {
                auto in_z = out_z + f_z - center_z;
                if (in_z < 0 || in_z > root_hi_z) continue;

                acc += in[in_x][in_y][in_z] *
                       filter[f_extent_x - f_x - 1][f_extent_y - f_y - 1][f_extent_z - f_z - 1];
              }
            }
          }
          out[out_x][out_y][out_z] = acc;
        }
  }
};

/*static*/ void ConvolveTask::cpu_variant(TaskContext& context)
{
  convolve_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { ConvolveTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
