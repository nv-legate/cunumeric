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
"""Consolidate test configuration from command-line and environment.

"""
from __future__ import annotations

import sys
from subprocess import CompletedProcess
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from .. import system as m


@pytest.fixture
def mock_subprocess_run(mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(m, "stdlib_run")


CMD = "legate script.py --cpus 4"


class TestSystem:
    def test_init(self) -> None:
        s = m.System()
        assert s.dry_run is False

    def test_run(self, mock_subprocess_run: MagicMock) -> None:
        s = m.System()

        expected = m.ProcessResult(CMD, returncode=10, output="<output>")
        mock_subprocess_run.return_value = CompletedProcess(
            CMD, 10, stdout="<output>"
        )

        result = s.run(CMD.split())
        mock_subprocess_run.assert_called()

        assert result == expected

    def test_dry_run(self, mock_subprocess_run: MagicMock) -> None:
        s = m.System(dry_run=True)

        result = s.run(CMD.split())
        mock_subprocess_run.assert_not_called()

        assert result.output == ""
        assert result.skipped

    def test_cpus(self) -> None:
        s = m.System()
        cpus = s.cpus
        assert len(cpus) > 0
        assert all(len(cpu.ids) > 0 for cpu in cpus)

    @pytest.mark.skipif(sys.platform != "linux", reason="pynvml required")
    def test_gpus(self) -> None:
        s = m.System()
        # can't really assume / test much here
        s.gpus
