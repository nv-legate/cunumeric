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

import multiprocessing
import sys

from tests._utils.config import Config
from tests._utils.system import System
from tests._utils.test_plan import TestPlan

if __name__ == "__main__":
    config = Config(sys.argv)

    manager = multiprocessing.Manager()

    system = System(manager=manager, dry_run=config.dry_run)

    plan = TestPlan(config, system)

    sys.exit(plan.execute())
