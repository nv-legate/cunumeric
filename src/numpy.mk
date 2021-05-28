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
GEN_CPU_SRC += ternary/where.cc               \
							 binary/binary_op.cc            \
							 binary/binary_op_util.cc       \
							 binary/scalar_binary_op.cc     \
							 binary/binary_red.cc           \
							 unary/scalar_unary_red.cc      \
							 unary/unary_op.cc              \
							 unary/unary_op_util.cc         \
							 unary/unary_red.cc             \
							 unary/unary_red_util.cc        \
							 unary/scalar_unary_op.cc       \
							 unary/convert.cc               \
							 unary/scalar_convert.cc        \
							 nullary/fill.cc                \
							 item/read.cc                   \
							 item/write.cc                  \
							 matrix/diag.cc                 \
							 matrix/matmul.cc               \
							 matrix/matvecmul.cc            \
							 matrix/dot.cc                  \
							 matrix/util.cc                 \
							 stat/bincount.cc               \
							 core.cc                        \
							 deserializer.cc                \
							 scalar.cc                      \
							 arange.cc                      \
							 arg.cc                         \
							 argmin.cc                      \
							 eye.cc                         \
							 mapper.cc                      \
							 nonzero.cc                     \
							 proj.cc                        \
							 rand.cc                        \
							 scan.cc                        \
							 shard.cc                       \
							 sort.cc                        \
							 tile.cc                        \
							 trans.cc                       \
							 numpy.cc # This must always be the last file!
                        # It guarantees we do our registration callback
                        # only after all task variants are recorded

ifeq ($(strip $(USE_OPENMP)),1)
GEN_CPU_SRC += ternary/where_omp.cc          \
							 binary/binary_op_omp.cc       \
							 binary/binary_red_omp.cc      \
							 unary/unary_op_omp.cc         \
							 unary/scalar_unary_red_omp.cc \
							 unary/unary_red_omp.cc        \
							 unary/convert_omp.cc          \
							 nullary/fill_omp.cc           \
							 matrix/diag_omp.cc            \
							 matrix/matmul_omp.cc          \
							 matrix/matvecmul_omp.cc       \
							 matrix/dot_omp.cc             \
							 stat/bincount_omp.cc
endif

GEN_GPU_SRC += ternary/where.cu               \
							 binary/binary_op.cu            \
							 binary/binary_red.cu           \
							 unary/scalar_unary_red.cu      \
							 unary/unary_red.cu             \
							 unary/unary_op.cu              \
							 unary/convert.cu               \
							 nullary/fill.cu                \
							 item/read.cu                   \
							 item/write.cu                  \
							 matrix/diag.cu                 \
							 matrix/matmul.cu               \
							 matrix/matvecmul.cu            \
							 matrix/dot.cu                  \
							 stat/bincount.cu               \
							 arange.cu                      \
							 arg.cu                         \
							 argmin.cu                      \
							 eye.cu                         \
							 nonzero.cu                     \
							 rand.cu                        \
							 scan.cu                        \
							 sort.cu                        \
							 tile.cu                        \
							 trans.cu
