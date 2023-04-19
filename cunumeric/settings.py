# Copyright 2023 NVIDIA Corporation
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
from __future__ import annotations

from legate.util.settings import (
    EnvOnlySetting,
    PrioritizedSetting,
    Settings,
    convert_bool,
    convert_int,
)

__all__ = ("settings",)


class CunumericRuntimeSettings(Settings):
    test: PrioritizedSetting[bool] = PrioritizedSetting(
        "test",
        "CUNUMERIC_TEST",
        default=False,
        convert=convert_bool,
        help="""
        Enable test mode. In test mode, all cuNumeric ndarrays are managed by
        the distributed runtime and the NumPy fallback for small arrays is
        turned off.
        """,
    )

    preload_cudalibs: PrioritizedSetting[bool] = PrioritizedSetting(
        "preload_cudalibs",
        "CUNUMERIC_PRELOAD_CUDALIBS",
        default=False,
        convert=convert_bool,
        help="""
        Preload and initialize handles of all CUDA libraries (cuBLAS, cuSOLVER,
        etc.) used in cuNumeric
        """,
    )

    warn: PrioritizedSetting[bool] = PrioritizedSetting(
        "warn",
        "CUNUMERIC_WARN",
        default=False,
        convert=convert_bool,
        help="""
        Turn on warnings
        """,
    )

    report_coverage: PrioritizedSetting[bool] = PrioritizedSetting(
        "report_coverage",
        "CUNUMERIC_REPORT_COVERAGE",
        default=False,
        convert=convert_bool,
        help="""
        Print an overall percentage of cunumeric coverage
        """,
    )

    report_dump_callstack: PrioritizedSetting[bool] = PrioritizedSetting(
        "report_dump_callstack",
        "CUNUMERIC_REPORT_DUMP_CALLSTACK",
        default=False,
        convert=convert_bool,
        help="""
        Print an overall percentage of cunumeric coverage with call stack info
        """,
    )

    report_dump_csv: PrioritizedSetting[str | None] = PrioritizedSetting(
        "report_dump_csv",
        "CUNUMERIC_REPORT_DUMP_CSV",
        default=None,
        help="""
        Save a coverage report to a specified CSV file
        """,
    )

    fast_math: EnvOnlySetting[int] = EnvOnlySetting(
        "fast_math",
        "CUNUMERIC_FAST_MATH",
        default=False,
        test_default=False,
        convert=convert_bool,
        help="""
        Whether to use fast-math.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_gpu_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_gpu_chunk",
        "CUNUMERIC_MIN_GPU_CHUNK",
        default=1048576,  # 1 << 20
        test_default=2,
        convert=convert_int,
        help="""
        If using GPUs, any task operating on arrays smaller than this will
        not be parallelized across more than one GPU.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_cpu_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_cpu_chunk",
        "CUNUMERIC_MIN_CPU_CHUNK",
        default=16384,  # 1 << 14
        test_default=2,
        convert=convert_int,
        help="""
        If using CPUs, any task operating on arrays smaller than this will
        not be parallelized across more than one core.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    min_omp_chunk: EnvOnlySetting[int] = EnvOnlySetting(
        "min_omp_chunk",
        "CUNUMERIC_MIN_OMP_CHUNK",
        default=131072,  # 1 << 17
        test_default=2,
        convert=convert_int,
        help="""
        If using OpenMP, any task operating on arrays smaller than this will
        not be parallelized across more than one OpenMP group.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    eager_fraction: EnvOnlySetting[int] = EnvOnlySetting(
        "eager_fraction",
        "CUNUMERIC_EAGER_FRACTION",
        default=16,
        test_default=1,
        convert=convert_int,
        help="""
        Eager fraction

        This is a read-only environment variable setting used by the runtime.
        """,
    )


settings = CunumericRuntimeSettings()
