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

GEN_CPU_SRC += cunumeric/random/bitgenerator.cc                            \
							 cunumeric/random/randutil/generator_host.cc                 \
							 cunumeric/random/randutil/generator_host_straightforward.cc

GEN_GPU_SRC += cunumeric/random/bitgenerator.cu                              \
							 cunumeric/random/randutil/generator_device.cu                 \
							 cunumeric/random/randutil/generator_device_straightforward.cu \