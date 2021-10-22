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

ifndef LEGATE_DIR
$(error LEGATE_DIR variable is not defined, aborting build)
endif
ifndef OPEN_BLAS_DIR
$(error OPEN_BLAS_DIR variable is not defined, aborting build)
endif

include $(LEGATE_DIR)/share/legate/config.mk

LIBNAME		= libcunumeric

CC_FLAGS ?=
CC_FLAGS += -I. -I$(OPEN_BLAS_DIR)/include
LD_FLAGS ?=
LD_FLAGS += $(OPENBLAS_FLAGS)
ifeq ($(strip $(USE_CUDA)),1)
LD_FLAGS += -lcufft
endif

NVCC_FLAGS ?=
NVCC_FLAGS += -I. -Wno-deprecated-declarations

CHECK_BOUNDS ?= 0
ifeq ($(strip $(CHECK_BOUNDS)),1)
CC_FLAGS += -DBOUNDS_CHECKS
endif

GEN_CPU_SRC	=
GEN_GPU_SRC	=

include cunumeric.mk

include $(LEGATE_DIR)/share/legate/legate.mk
