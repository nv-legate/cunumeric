/* Copyright 2022 NVIDIA Corporation
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

#include "generator.cuh"

#include "generator_beta.inl"
template struct randutilimpl::dispatcher<randutilimpl::execlocation::DEVICE, beta_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, beta_t<double>, double>;

#include "generator_f.inl"
template struct randutilimpl::dispatcher<randutilimpl::execlocation::DEVICE, f_t<float>, float>;
template struct randutilimpl::dispatcher<randutilimpl::execlocation::DEVICE, f_t<double>, double>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, noncentralf_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, noncentralf_t<double>, double>;

#include "generator_logseries.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, logseries_t<double>, uint32_t>;