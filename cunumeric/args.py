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

from legate.util.args import ArgSpec, Argument

ARGS = [
    Argument(
        "test",
        ArgSpec(
            action="store_true",
            default=False,
            dest="test_mode",
            help="Enable test mode. In test mode, all cuNumeric ndarrays are managed by the distributed runtime and the NumPy fallback for small arrays is turned off.",  # noqa E501
        ),
    ),
    Argument(
        "preload-cudalibs",
        ArgSpec(
            action="store_true",
            default=False,
            dest="preload_cudalibs",
            help="Preload and initialize handles of all CUDA libraries (cuBLAS, cuSOLVER, etc.) used in cuNumericLoad CUDA libs early",  # noqa E501
        ),
    ),
    Argument(
        "warn",
        ArgSpec(
            action="store_true",
            default=False,
            dest="warning",
            help="Turn on warnings",
        ),
    ),
    Argument(
        "report:coverage",
        ArgSpec(
            action="store_true",
            default=False,
            dest="report_coverage",
            help="Print an overall percentage of cunumeric coverage",
        ),
    ),
    Argument(
        "report:dump-callstack",
        ArgSpec(
            action="store_true",
            default=False,
            dest="report_dump_callstack",
            help="Print an overall percentage of cunumeric coverage with call stack details",  # noqa E501
        ),
    ),
    Argument(
        "report:dump-csv",
        ArgSpec(
            action="store",
            type=str,
            nargs="?",
            default=None,
            dest="report_dump_csv",
            help="Save a coverage report to a specified CSV file",
        ),
    ),
]
