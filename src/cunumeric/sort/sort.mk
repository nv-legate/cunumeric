# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

GEN_CPU_SRC += cunumeric/sort/sort.cc

ifeq ($(strip $(USE_OPENMP)),1)
GEN_CPU_SRC += cunumeric/sort/sort_omp.cc
endif

GEN_GPU_SRC += cunumeric/sort/sort.cu                   \
							 cunumeric/sort/cub_sort_bool.cu          \
							 cunumeric/sort/cub_sort_int8.cu          \
							 cunumeric/sort/cub_sort_int16.cu         \
							 cunumeric/sort/cub_sort_int32.cu         \
							 cunumeric/sort/cub_sort_int64.cu         \
							 cunumeric/sort/cub_sort_uint8.cu         \
							 cunumeric/sort/cub_sort_uint16.cu        \
							 cunumeric/sort/cub_sort_uint32.cu        \
							 cunumeric/sort/cub_sort_uint64.cu        \
							 cunumeric/sort/cub_sort_half.cu          \
							 cunumeric/sort/cub_sort_float.cu         \
							 cunumeric/sort/cub_sort_double.cu        \
							 cunumeric/sort/thrust_sort_bool.cu       \
							 cunumeric/sort/thrust_sort_int8.cu       \
							 cunumeric/sort/thrust_sort_int16.cu      \
							 cunumeric/sort/thrust_sort_int32.cu      \
							 cunumeric/sort/thrust_sort_int64.cu      \
							 cunumeric/sort/thrust_sort_uint8.cu      \
							 cunumeric/sort/thrust_sort_uint16.cu     \
							 cunumeric/sort/thrust_sort_uint32.cu     \
							 cunumeric/sort/thrust_sort_uint64.cu     \
							 cunumeric/sort/thrust_sort_half.cu       \
							 cunumeric/sort/thrust_sort_float.cu      \
							 cunumeric/sort/thrust_sort_double.cu     \
							 cunumeric/sort/thrust_sort_complex64.cu  \
							 cunumeric/sort/thrust_sort_complex128.cu
