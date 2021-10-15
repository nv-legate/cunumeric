# Copyright 2021 NVIDIA Corporation
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
GEN_CPU_SRC += numpy/ternary/where.cc               \
							 numpy/binary/binary_op.cc            \
							 numpy/binary/binary_red.cc           \
							 numpy/unary/scalar_unary_red.cc      \
							 numpy/unary/unary_op.cc              \
							 numpy/unary/unary_red.cc             \
							 numpy/unary/convert.cc               \
							 numpy/nullary/arange.cc              \
							 numpy/nullary/eye.cc                 \
							 numpy/nullary/fill.cc                \
							 numpy/item/read.cc                   \
							 numpy/item/write.cc                  \
							 numpy/matrix/diag.cc                 \
							 numpy/matrix/matmul.cc               \
							 numpy/matrix/matvecmul.cc            \
							 numpy/matrix/dot.cc                  \
							 numpy/matrix/tile.cc                 \
							 numpy/matrix/transpose.cc            \
							 numpy/matrix/util.cc                 \
							 numpy/random/rand.cc                 \
							 numpy/search/nonzero.cc              \
							 numpy/stat/bincount.cc               \
							 numpy/convolution/convolve.cc        \
							 numpy/transform/flip.cc              \
							 numpy/arg.cc                         \
							 numpy/mapper.cc                      \
							 numpy/numpy.cc # This must always be the last file!
                              # It guarantees we do our registration callback
                              # only after all task variants are recorded

ifeq ($(strip $(USE_OPENMP)),1)
GEN_CPU_SRC += numpy/ternary/where_omp.cc          \
							 numpy/binary/binary_op_omp.cc       \
							 numpy/binary/binary_red_omp.cc      \
							 numpy/unary/unary_op_omp.cc         \
							 numpy/unary/scalar_unary_red_omp.cc \
							 numpy/unary/unary_red_omp.cc        \
							 numpy/unary/convert_omp.cc          \
							 numpy/nullary/arange_omp.cc         \
							 numpy/nullary/eye_omp.cc            \
							 numpy/nullary/fill_omp.cc           \
							 numpy/matrix/diag_omp.cc            \
							 numpy/matrix/matmul_omp.cc          \
							 numpy/matrix/matvecmul_omp.cc       \
							 numpy/matrix/dot_omp.cc             \
							 numpy/matrix/tile_omp.cc            \
							 numpy/matrix/transpose_omp.cc       \
							 numpy/matrix/util_omp.cc            \
							 numpy/random/rand_omp.cc            \
							 numpy/search/nonzero_omp.cc         \
							 numpy/stat/bincount_omp.cc          \
							 numpy/transform/flip_omp.cc
endif

GEN_GPU_SRC += numpy/ternary/where.cu               \
							 numpy/binary/binary_op.cu            \
							 numpy/binary/binary_red.cu           \
							 numpy/unary/scalar_unary_red.cu      \
							 numpy/unary/unary_red.cu             \
							 numpy/unary/unary_op.cu              \
							 numpy/unary/convert.cu               \
							 numpy/nullary/arange.cu              \
							 numpy/nullary/eye.cu                 \
							 numpy/nullary/fill.cu                \
							 numpy/item/read.cu                   \
							 numpy/item/write.cu                  \
							 numpy/matrix/diag.cu                 \
							 numpy/matrix/matmul.cu               \
							 numpy/matrix/matvecmul.cu            \
							 numpy/matrix/dot.cu                  \
							 numpy/matrix/tile.cu                 \
							 numpy/matrix/transpose.cu            \
							 numpy/random/rand.cu                 \
							 numpy/search/nonzero.cu              \
							 numpy/stat/bincount.cu               \
							 numpy/convolution/convolve.cu	      \
							 numpy/transform/flip.cu              \
							 numpy/numpy.cu
