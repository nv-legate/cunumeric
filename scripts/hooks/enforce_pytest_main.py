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
import sys
from pathlib import Path
from typing import NoReturn

BOILERPLATE = "sys.exit(pytest.main(sys.argv))"

TESTS_TOP = Path(__file__).parents[2] / "tests"

TEST_DIRS = (TESTS_TOP / "unit", TESTS_TOP / "integration")


def enforce_pytest_main() -> NoReturn:
    for dir in TEST_DIRS:
        for path in dir.rglob("*.py"):
            if not path.is_file() or not path.name.startswith("test_"):
                continue

            if not open(path).readlines()[-1].strip() == BOILERPLATE:
                print(f"Test file {path} missing required boilerplate")
                sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    enforce_pytest_main()
