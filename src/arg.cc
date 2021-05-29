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

#include "arg.h"

using namespace Legion;

namespace legate {
namespace numpy {

#define DECLARE_ARGMAX_IDENTITY(TYPE) \
  template <>                         \
  const Argval<TYPE> ArgmaxReduction<TYPE>::identity = Argval<TYPE>(MaxReduction<TYPE>::identity);

#define DECLARE_ARGMIN_IDENTITY(TYPE) \
  template <>                         \
  const Argval<TYPE> ArgminReduction<TYPE>::identity = Argval<TYPE>(MinReduction<TYPE>::identity);

#define DECLARE_IDENTITIES(TYPE) \
  DECLARE_ARGMAX_IDENTITY(TYPE)  \
  DECLARE_ARGMIN_IDENTITY(TYPE)

DECLARE_IDENTITIES(__half)
DECLARE_IDENTITIES(float)
DECLARE_IDENTITIES(double)
DECLARE_IDENTITIES(bool)
DECLARE_IDENTITIES(int8_t)
DECLARE_IDENTITIES(int16_t)
DECLARE_IDENTITIES(int32_t)
DECLARE_IDENTITIES(int64_t)
DECLARE_IDENTITIES(uint8_t)
DECLARE_IDENTITIES(uint16_t)
DECLARE_IDENTITIES(uint32_t)
DECLARE_IDENTITIES(uint64_t)
DECLARE_IDENTITIES(complex<float>)

}  // namespace numpy
}  // namespace legate
