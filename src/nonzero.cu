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

#include "cuda_help.h"
#include "fill.cuh"
#include "nonzero.h"
#include "proj.h"
#include <cstdio>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>
#include <utility>

using namespace Legion;

namespace legate {
namespace numpy {

namespace detail {

template <typename Tuple, size_t... Is>
__device__ __host__ auto make_thrust_tuple_impl(Tuple&& t, std::index_sequence<Is...>)
{
  return thrust::make_tuple(std::get<Is>(std::forward<Tuple>(t))...);
}

template <typename Tuple>
__device__ __host__ auto make_thrust_tuple(Tuple&& t)
{
  using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
  return make_thrust_tuple_impl(std::forward<Tuple>(t), Indices{});
}

template <size_t dims, typename Rect>
__device__ __host__ size_t volume(const Rect rect, size_t level)
{
  size_t ret = 1;
  for (auto i = level; i < dims; ++i) { ret *= rect.hi[i] - rect.lo[i] + 1; }
  return ret;
}

template <size_t dims, size_t level>
struct helper {
  template <typename Accessor, typename Rect>
  __device__ __host__ decltype(auto) operator()(Accessor accessor,
                                                const Rect rect,
                                                const coord_t position) const
  {
    const size_t vol    = volume<dims>(rect, level);
    const size_t offset = position / vol;
    const coord_t index = rect.lo[level - 1] + offset;
    // cudaDeviceSynchronize();
    return helper<dims, level + 1>{}(accessor[index], rect, position - offset * vol);
  }
};

template <size_t dims>
struct helper<dims, dims> {
  template <typename Accessor, typename Rect>
  __device__ __host__ decltype(auto) operator()(Accessor accessor,
                                                const Rect rect,
                                                const coord_t position) const
  {
    // cudaDeviceSynchronize();
    return accessor[static_cast<coord_t>(rect.lo[dims - 1] + position % (rect.hi[dims - 1] + 1))];
  }
};

template <size_t dims>
struct accessor_helper {
  template <typename Accessor, typename Rect>
  __device__ __host__ decltype(auto) operator()(Accessor accessor,
                                                const Rect rect,
                                                const coord_t position) const
  {
    // cudaDeviceSynchronize();
    return helper<dims, 1>{}(accessor, rect, position);
  }
};

template <size_t dims, size_t level>
struct ihelper {
  template <typename Rect>
  __device__ __host__ decltype(auto) operator()(const Rect rect, const coord_t position) const
  {
    const size_t vol    = volume<dims>(rect, level);
    const size_t offset = position / vol;
    const coord_t index = rect.lo[level - 1] + offset;
    // cudaDeviceSynchronize();
    return std::tuple_cat(std::make_tuple(index),
                          ihelper<dims, level + 1>{}(rect, position - offset * vol));
  }
};

template <size_t dims>
struct ihelper<dims, dims> {
  template <typename Rect>
  __device__ __host__ decltype(auto) operator()(const Rect rect, const coord_t position) const
  {
    // cudaDeviceSynchronize();
    return std::make_tuple(
      static_cast<coord_t>(rect.lo[dims - 1] + position % (rect.hi[dims - 1] + 1)));
  }
};

template <typename Rect, size_t dims>
struct index_helper {
  __device__ __host__ decltype(auto) operator()(const Rect rect, const coord_t position) const
  {
    // cudaDeviceSynchronize();
    return make_thrust_tuple(ihelper<dims, 1>{}(rect, position));
  }
};

// template<typename Accessor, typename Rect>
// struct accessor_helper<Accessor, Rect, 1> {
//   __device__ __host__ decltype(auto) operator()(const Accessor& accessor, const Rect& rect, const
//   coord_t position) const {
//     printf("position: %llu, rect.lo[0]: %llu, rect.hi[0]: %llu\n", position, rect.lo[0],
//     rect.hi[0]); printf("Accessing at %llu \n", rect.lo[0] + position); return
//     accessor[rect.lo[0] + position];
//   }

//   // __device__ __host__ decltype(auto) operator()(Accessor& accessor, Rect& rect, coord_t
//   position) const {
//   //   printf("position: %llu, rect.lo[0]: %llu, rect.hi[0]: %llu\n", position, rect.lo[0],
//   rect.hi[0]);
//   //   printf("Accessing at %llu \n", rect.lo[0] + position);
//   //   return accessor[rect.lo[0] + position];
//   // }
// };

// template<typename Accessor, typename Rect>
// struct accessor_helper<Accessor, Rect, 2> {
//   __device__ __host__ decltype(auto) operator()(const Accessor& accessor, const Rect& rect,
//   coord_t position) const {
//     printf("position: %llu, rect.lo[0]: %llu, rect.hi[0]: %llu, rect.lo[1]: %llu, rect.hi[1]:
//     %llu\n", position, rect.lo[0],
//            rect.hi[0], rect.lo[1], rect.hi[1]);
//     printf("Accessing at %llu, %llu \n", rect.lo[0] + position / (rect.hi[1] + 1), rect.lo[1] +
//     position % (rect.hi[1] + 1)); return accessor[rect.lo[0] + position / (rect.hi[1] +
//     1)][rect.lo[1] + position % (rect.hi[1] + 1)];
//   }

//   // __device__ __host__ decltype(auto) operator()(Accessor& accessor, Rect& rect, coord_t
//   position) const {
//   //   printf("position: %llu, rect.lo[0]: %llu, rect.hi[0]: %llu, rect.lo[1]: %llu, rect.hi[1]:
//   %llu\n", position, rect.lo[0],
//   //          rect.hi[0], rect.lo[1], rect.hi[1]);
//   //   printf("Accessing at %llu, %llu \n", rect.lo[0] + position / (rect.hi[1] + 1), rect.lo[1]
//   + position % (rect.hi[1] + 1));
//   //   return accessor[rect.lo[0] + position / (rect.hi[1] + 1)][rect.lo[1] + position %
//   (rect.hi[1] + 1)];
//   // }
// };
}  // namespace detail

template <typename Rect, size_t dim = Rect::dim, bool row_major = true>
class index_iterator
  : public thrust::iterator_facade<
      index_iterator<Rect, dim, row_major>,
      std::remove_reference_t<decltype(detail::index_helper<Rect, dim>{}(std::declval<Rect>(), 0))>,
      thrust::any_system_tag,
      thrust::random_access_traversal_tag,
      decltype(detail::index_helper<Rect, dim>{}(std::declval<Rect>(), 0)),
      std::ptrdiff_t> {
 public:
  index_iterator(const Rect rect, const coord_t position) : rect{rect}, position{position} {}

 private:
  friend class thrust::iterator_core_access;

  __device__ __host__ decltype(auto) dereference() const
  {
    return detail::index_helper<Rect, dim>{}(rect, position);
  }

  __device__ __host__ bool equal(const index_iterator& rhs) const
  {
    return rhs.position == position;
  }

  __device__ __host__ void increment() { position++; }

  __device__ __host__ void advance(std::ptrdiff_t diff) { position += diff; }

  __device__ __host__ std::ptrdiff_t distance_to(const index_iterator& rhs) const
  {
    return rhs.position - position;
  }

  coord_t position{0};
  Rect rect;

  // Need to implement column major
  static_assert(row_major == true, "column major not implemented");
};

template <size_t dim, typename Rect, bool row_major = true>
index_iterator<Rect, dim, row_major> make_index_iterator(const Rect rect, coord_t position)
{
  return index_iterator<Rect, dim, row_major>(rect, position);
}

template <typename Accessor,
          typename Rect,
          size_t dim     = Accessor::dim,
          typename T     = typename Accessor::value_type,
          bool row_major = true>
class accessor_iterator
  : public thrust::iterator_facade<accessor_iterator<Accessor, Rect, dim, T, row_major>,
                                   T,
                                   thrust::any_system_tag,
                                   thrust::random_access_traversal_tag,
                                   decltype(detail::accessor_helper<dim>{}(
                                     std::declval<Accessor>(), std::declval<Rect>(), 0)),
                                   std::ptrdiff_t> {
 public:
  accessor_iterator(const Accessor accessor, const Rect rect) : accessor{accessor}, rect{rect} {}

 private:
  friend class thrust::iterator_core_access;

  __device__ __host__ decltype(auto) dereference() const
  {
    return detail::accessor_helper<dim>{}(accessor, rect, position);
  }

  __device__ __host__ bool equal(const accessor_iterator& rhs) const
  {
    return rhs.position == position;
  }

  __device__ __host__ void increment() { position++; }

  __device__ __host__ void advance(std::ptrdiff_t diff) { position += diff; }

  __device__ __host__ std::ptrdiff_t distance_to(const accessor_iterator& rhs) const
  {
    return rhs.position - position;
  }

  Accessor accessor;
  coord_t position{0};
  Rect rect;

  // Need to implement column major
  static_assert(row_major == true, "column major not implemented");
};

template <typename Accessor, typename Rect>
accessor_iterator<Accessor, Rect> make_accessor_iterator(const Accessor acc, const Rect rect)
{
  return accessor_iterator<Accessor, Rect>(acc, rect);
}

template <typename T>
/*static*/ void NonzeroTask<T>::gpu_variant(const Task* task,
                                            const std::vector<PhysicalRegion>& regions,
                                            Context ctx,
                                            Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const int in_dim = derez.unpack_dimension();
  assert(in_dim > 0);
  switch (in_dim) {
    case 1: {
      const Rect<1> in_rect     = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
      const AccessorRO<T, 1> in = derez.unpack_accessor_RO<T, 1>(regions[0], in_rect);
      const Rect<2> out_rect    = regions[1];
      const AccessorRW<uint64_t, 2> out =
        derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      auto in_iter  = make_accessor_iterator(in, in_rect);
      auto out_iter = make_accessor_iterator(out, out_rect);

      thrust::copy_if(thrust::device,
                      thrust::make_counting_iterator(in_rect.lo[0]),
                      thrust::make_counting_iterator(in_rect.hi[0] + 1),
                      in_iter,
                      out_iter,
                      [] __CUDA_HD__(const T& arg) { return arg != T{0}; });

      break;
    }
    case 2: {
      const Rect<2> in_rect     = NumPyProjectionFunctor::unpack_shape<2>(task, derez);
      const AccessorRO<T, 2> in = derez.unpack_accessor_RO<T, 2>(regions[0], in_rect);
      const Rect<2> out_rect    = regions[1];
      const AccessorRW<uint64_t, 2> out =
        derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      auto in_iter  = make_accessor_iterator(in, in_rect);
      auto out_iter = make_accessor_iterator(out, out_rect);

      thrust::copy_if(thrust::device,
                      make_index_iterator<2>(in_rect, 0),
                      make_index_iterator<2>(in_rect, in_rect.volume()),
                      in_iter,
                      thrust::make_zip_iterator(
                        thrust::make_tuple(out_iter, out_iter + detail::volume<2>(out_rect, 1))),
                      [] __CUDA_HD__(const T& arg) { return arg != T{0}; });

      break;
    }
    case 3: {
      const Rect<3> in_rect     = NumPyProjectionFunctor::unpack_shape<3>(task, derez);
      const AccessorRO<T, 3> in = derez.unpack_accessor_RO<T, 3>(regions[0], in_rect);
      const Rect<2> out_rect    = regions[1];
      const AccessorRW<uint64_t, 2> out =
        derez.unpack_accessor_RW<uint64_t, 2>(regions[1], out_rect);
      auto in_iter  = make_accessor_iterator(in, in_rect);
      auto out_iter = make_accessor_iterator(out, out_rect);

      thrust::copy_if(thrust::device,
                      make_index_iterator<3>(in_rect, 0),
                      make_index_iterator<3>(in_rect, in_rect.volume()),
                      in_iter,
                      thrust::make_zip_iterator(
                        thrust::make_tuple(out_iter,
                                           out_iter + detail::volume<2>(out_rect, 1),
                                           out_iter + 2 * detail::volume<2>(out_rect, 1))),
                      [] __CUDA_HD__(const T& arg) { return arg != T{0}; });

      break;
    }
    default: assert(false);
  }
}

INSTANTIATE_TASK_VARIANT(NonzeroTask, gpu_variant)

namespace detail {
struct MakeRect {
  __CUDA_HD__ MakeRect(coord_t nonzero_dim) : nonzero_dim{nonzero_dim} {}

  template <typename T>
  __CUDA_HD__ Rect<2> operator()(const T lo, const T hi)
  {
    coord_t pt1[2] = {0, static_cast<coord_t>(lo)};
    coord_t pt2[2] = {nonzero_dim, static_cast<coord_t>(hi) - 1};
    return Rect<2>{Point<2>(pt1), Point<2>(pt2)};
  }

  coord_t nonzero_dim;
};
}  // namespace detail

template <typename T>
/*static*/ void ConvertRangeToRectTask<T>::gpu_variant(const Task* task,
                                                       const std::vector<PhysicalRegion>& regions,
                                                       Context ctx,
                                                       Runtime* runtime)
{
  LegateDeserializer derez(task->args, task->arglen);
  const coord_t nonzero_dim = derez.unpack_32bit_int();
  const int dim             = derez.unpack_dimension();
  assert(dim == 1);
  const Rect<1> rect = NumPyProjectionFunctor::unpack_shape<1>(task, derez);
  assert(!rect.empty());
  const auto begin                 = rect.lo[0];
  const auto end                   = rect.hi[0];
  const AccessorRO<T, 1> in        = derez.unpack_accessor_RO<T, 1>(regions[0], rect);
  const AccessorRW<Rect<2>, 1> out = derez.unpack_accessor_RW<Rect<2>, 1>(regions[1], rect);
  auto in_iter                     = make_accessor_iterator(in, rect);
  auto out_iter                    = make_accessor_iterator(out, rect);
  auto const size                  = rect.volume();

  Rect<1> bounds(Point<1>(0), Point<1>(size + 1));
  auto buffer = DeferredBuffer<T, 1>(Memory::GPU_FB_MEM, Domain(bounds));
  thrust::uninitialized_fill(thrust::device, out_iter, out_iter + size, Rect<2>{});
  // We just really need to initialize the first value to 0, but uninitialized fill of the whole
  // buffer is just easier to do
  thrust::uninitialized_fill(thrust::device, buffer.ptr(0), buffer.ptr(0) + size + 1, 0);
  thrust::copy(thrust::device, in_iter, in_iter + size, buffer.ptr(0) + 1);
  thrust::transform(thrust::device,
                    buffer.ptr(0),
                    buffer.ptr(0) + size,
                    buffer.ptr(0) + 1,
                    out_iter,
                    detail::MakeRect{nonzero_dim});
}

INSTANTIATE_INT_VARIANT(ConvertRangeToRectTask, gpu_variant)
INSTANTIATE_UINT_VARIANT(ConvertRangeToRectTask, gpu_variant)

}  // namespace numpy
}  // namespace legate
