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

import numpy as np
import pytest
from legate.util.settings import PrioritizedSetting

import cunumeric.settings as m

_expected_settings = (
    "test",
    "preload_cudalibs",
    "warn",
    "report_coverage",
    "report_dump_callstack",
    "report_dump_csv",
)


class TestSettings:
    def test_standard_settings(self) -> None:
        settings = [
            k
            for k, v in m.settings.__class__.__dict__.items()
            if isinstance(v, PrioritizedSetting)
        ]
        assert set(settings) == set(_expected_settings)

    @pytest.mark.parametrize("name", _expected_settings)
    def test_prefix(self, name: str) -> None:
        ps = getattr(m.settings, name)
        assert ps.env_var.startswith("CUNUMERIC_")

    @pytest.mark.parametrize("name", _expected_settings)
    def test_parent(self, name: str) -> None:
        ps = getattr(m.settings, name)
        assert ps._parent == m.settings

    def test_types(self) -> None:
        assert m.settings.test.convert_type == "bool"
        assert m.settings.preload_cudalibs.convert_type == "bool"
        assert m.settings.warn.convert_type == "bool"
        assert m.settings.report_coverage.convert_type == "bool"
        assert m.settings.report_dump_callstack.convert_type == "bool"
        assert m.settings.report_dump_csv.convert_type == "str"


class TestDefaults:
    def test_test(self) -> None:
        assert m.settings.test.default is False

    def test_preload_cudalibs(self) -> None:
        assert m.settings.preload_cudalibs.default is False

    def test_warn(self) -> None:
        assert m.settings.warn.default is False

    def test_report_coverage(self) -> None:
        assert m.settings.report_coverage.default is False

    def test_report_dump_callstack(self) -> None:
        assert m.settings.report_dump_callstack.default is False

    def test_report_dump_csv(self) -> None:
        assert m.settings.report_dump_csv.default is None


if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
