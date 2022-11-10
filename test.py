#!/usr/bin/env python

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
from __future__ import annotations

import sys

from legate.tester import PER_FILE_ARGS, SKIPPED_EXAMPLES
from legate.tester.config import Config
from legate.tester.test_plan import TestPlan
from legate.tester.test_system import TestSystem

SKIPPED_EXAMPLES.update(
    {
        "examples/ingest.py",
        "examples/kmeans_sort.py",
        "examples/lstm_full.py",
        "examples/wgrad.py",
    }
)

PER_FILE_ARGS.update(
    {
        "examples/lstm_full.py": ["--file", "resources/lstm_input.txt"],
    }
)


if __name__ == "__main__":
    config = Config(sys.argv)

    system = TestSystem(dry_run=config.dry_run)

    plan = TestPlan(config, system)

    sys.exit(plan.execute())
