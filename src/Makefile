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

ifndef LEGATE_DIR
$(error LEGATE_DIR variable is not defined, aborting build)
endif
ifndef OPENBLAS_PATH
$(error OPENBLAS_PATH variable is not defined, aborting build)
endif
ifndef OPENBLAS_LIBNAME
$(error OPENBLAS_PATH variable is not defined, aborting build)
endif
ifndef TBLIS_PATH
$(error TBLIS_PATH variable is not defined, aborting build)
endif
ifeq ($(strip $(USE_CUDA)),1)
ifndef CUTENSOR_PATH
$(error CUTENSOR_PATH variable is not defined, aborting build)
endif
ifndef NCCL_PATH
$(error NCCL_PATH variable is not defined, aborting build)
endif
endif # ifeq ($(strip $(USE_CUDA)),1)
ifndef THRUST_PATH
$(error THRUST_PATH variable is not defined, aborting build)
endif

include $(LEGATE_DIR)/share/legate/config.mk

LIBNAME = libcunumeric

CC_FLAGS ?=
CC_FLAGS += -I. -I$(OPENBLAS_PATH)/include -I$(TBLIS_PATH)/include -I$(THRUST_PATH)
CC_FLAGS += -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP

LD_FLAGS ?=
LD_FLAGS += -L$(OPENBLAS_PATH)/lib -l$(OPENBLAS_LIBNAME) -Wl,-rpath,$(OPENBLAS_PATH)/lib
LD_FLAGS += -L$(TBLIS_PATH)/lib -ltblis -Wl,-rpath,$(TBLIS_PATH)/lib
ifeq ($(strip $(USE_CUDA)),1)
DEVICE_LD_FLAGS += -lcufft_static
LD_FLAGS += -lcublas -lcusolver -lcufft_static -lculibos
LD_FLAGS += -L$(CUTENSOR_PATH)/lib -lcutensor -Wl,-rpath,$(CUTENSOR_PATH)/lib
LD_FLAGS += -L$(NCCL_PATH)/lib -lnccl -Wl,-rpath,$(NCCL_PATH)/lib
endif
NVCC_FLAGS ?=
NVCC_FLAGS += -I. -I$(CUTENSOR_PATH)/include -I$(NCCL_PATH)/include -I$(THRUST_PATH) -Wno-deprecated-declarations

ifeq ($(strip $(DEBUG)),1)
CC_FLAGS += -DDEBUG_CUNUMERIC
NVCC_FLAGS += -DDEBUG_CUNUMERIC
endif

CHECK_BOUNDS ?= 0
ifeq ($(strip $(CHECK_BOUNDS)),1)
CC_FLAGS += -DBOUNDS_CHECKS
endif

GEN_CPU_SRC =
GEN_GPU_SRC =

include cunumeric.mk

include $(LEGATE_DIR)/share/legate/legate.mk
