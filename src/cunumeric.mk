# Copyright 2021-2022 NVIDIA Corporation
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

# List all the application source files that need OpenMP separately
# since we have to add the -fopenmp flag to  CC_FLAGS for them
GEN_CPU_SRC += cunumeric/ternary/where.cc               \
							 cunumeric/binary/binary_op.cc            \
							 cunumeric/binary/binary_red.cc           \
							 cunumeric/unary/scalar_unary_red.cc      \
							 cunumeric/unary/unary_op.cc              \
							 cunumeric/unary/unary_red.cc             \
							 cunumeric/unary/convert.cc               \
							 cunumeric/nullary/arange.cc              \
							 cunumeric/nullary/eye.cc                 \
							 cunumeric/nullary/fill.cc                \
							 cunumeric/index/choose.cc                \
							 cunumeric/index/repeat.cc                \
							 cunumeric/item/read.cc                   \
							 cunumeric/item/write.cc                  \
							 cunumeric/matrix/contract.cc             \
							 cunumeric/matrix/diag.cc                 \
							 cunumeric/matrix/gemm.cc                 \
							 cunumeric/matrix/matmul.cc               \
							 cunumeric/matrix/matvecmul.cc            \
							 cunumeric/matrix/dot.cc                  \
							 cunumeric/matrix/potrf.cc                \
							 cunumeric/matrix/syrk.cc                 \
							 cunumeric/matrix/tile.cc                 \
							 cunumeric/matrix/transpose.cc            \
							 cunumeric/matrix/trilu.cc                \
							 cunumeric/matrix/trsm.cc                 \
							 cunumeric/matrix/util.cc                 \
							 cunumeric/random/rand.cc                 \
							 cunumeric/search/nonzero.cc              \
							 cunumeric/set/unique.cc                  \
							 cunumeric/set/unique_reduce.cc           \
							 cunumeric/sort/sort.cc                   \
							 cunumeric/stat/bincount.cc               \
							 cunumeric/convolution/convolve.cc        \
							 cunumeric/fft/fft.cc                     \
							 cunumeric/transform/flip.cc              \
							 cunumeric/arg.cc                         \
							 cunumeric/mapper.cc

ifeq ($(strip $(USE_OPENMP)),1)
GEN_CPU_SRC += cunumeric/ternary/where_omp.cc          \
							 cunumeric/binary/binary_op_omp.cc       \
							 cunumeric/binary/binary_red_omp.cc      \
							 cunumeric/unary/unary_op_omp.cc         \
							 cunumeric/unary/scalar_unary_red_omp.cc \
							 cunumeric/unary/unary_red_omp.cc        \
							 cunumeric/unary/convert_omp.cc          \
							 cunumeric/nullary/arange_omp.cc         \
							 cunumeric/nullary/eye_omp.cc            \
							 cunumeric/nullary/fill_omp.cc           \
							 cunumeric/index/choose_omp.cc           \
							 cunumeric/index/repeat_omp.cc           \
							 cunumeric/matrix/contract_omp.cc        \
							 cunumeric/matrix/diag_omp.cc            \
							 cunumeric/matrix/gemm_omp.cc            \
							 cunumeric/matrix/matmul_omp.cc          \
							 cunumeric/matrix/matvecmul_omp.cc       \
							 cunumeric/matrix/dot_omp.cc             \
							 cunumeric/matrix/potrf_omp.cc           \
							 cunumeric/matrix/syrk_omp.cc            \
							 cunumeric/matrix/tile_omp.cc            \
							 cunumeric/matrix/transpose_omp.cc       \
							 cunumeric/matrix/trilu_omp.cc           \
							 cunumeric/matrix/trsm_omp.cc            \
							 cunumeric/matrix/util_omp.cc            \
							 cunumeric/random/rand_omp.cc            \
							 cunumeric/search/nonzero_omp.cc         \
							 cunumeric/set/unique_omp.cc             \
							 cunumeric/sort/sort_omp.cc              \
							 cunumeric/stat/bincount_omp.cc          \
							 cunumeric/convolution/convolve_omp.cc   \
							 cunumeric/fft/fft_omp.cc                \
							 cunumeric/transform/flip_omp.cc
endif

GEN_CPU_SRC += cunumeric/cunumeric.cc # This must always be the last file!
                                      # It guarantees we do our registration callback
                                      # only after all task variants are recorded

GEN_GPU_SRC += cunumeric/ternary/where.cu               \
							 cunumeric/binary/binary_op.cu            \
							 cunumeric/binary/binary_red.cu           \
							 cunumeric/unary/scalar_unary_red.cu      \
							 cunumeric/unary/unary_red.cu             \
							 cunumeric/unary/unary_op.cu              \
							 cunumeric/unary/convert.cu               \
							 cunumeric/nullary/arange.cu              \
							 cunumeric/nullary/eye.cu                 \
							 cunumeric/nullary/fill.cu                \
							 cunumeric/item/read.cu                   \
							 cunumeric/item/write.cu                  \
							 cunumeric/index/choose.cu                \
							 cunumeric/index/repeat.cu                \
							 cunumeric/matrix/contract.cu             \
							 cunumeric/matrix/diag.cu                 \
							 cunumeric/matrix/gemm.cu                 \
							 cunumeric/matrix/matmul.cu               \
							 cunumeric/matrix/matvecmul.cu            \
							 cunumeric/matrix/dot.cu                  \
							 cunumeric/matrix/potrf.cu                \
							 cunumeric/matrix/syrk.cu                 \
							 cunumeric/matrix/tile.cu                 \
							 cunumeric/matrix/transpose.cu            \
							 cunumeric/matrix/trilu.cu                \
							 cunumeric/matrix/trsm.cu                 \
							 cunumeric/random/rand.cu                 \
							 cunumeric/search/nonzero.cu              \
							 cunumeric/set/unique.cu                  \
							 cunumeric/sort/sort.cu                   \
							 cunumeric/stat/bincount.cu               \
							 cunumeric/convolution/convolve.cu        \
               cunumeric/fft/fft.cu                     \
							 cunumeric/transform/flip.cu              \
							 cunumeric/cudalibs.cu                    \
							 cunumeric/cunumeric.cu

GEN_DEVICE_SRC += cunumeric/convolution/convolve_callbacks.cu
