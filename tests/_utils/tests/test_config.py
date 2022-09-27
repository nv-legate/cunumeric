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

from pathlib import Path, PurePath

import pytest

from .. import (
    DEFAULT_CPUS_PER_NODE,
    DEFAULT_GPU_DELAY,
    DEFAULT_GPU_MEMORY_BUDGET,
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_OMPS_PER_NODE,
    DEFAULT_OMPTHREADS,
    FEATURES,
    config as m,
)
from ..args import PIN_OPTIONS, PinOptionsType


class TestConfig:
    def test_default_init(self) -> None:
        c = m.Config([])

        assert c.examples is True
        assert c.integration is True
        assert c.unit is False
        assert c.files is None

        assert c.features == ("cpus",)

        assert c.cpus == DEFAULT_CPUS_PER_NODE
        assert c.gpus == DEFAULT_GPUS_PER_NODE
        assert c.cpu_pin == "partial"
        assert c.gpu_delay == DEFAULT_GPU_DELAY
        assert c.fbmem == DEFAULT_GPU_MEMORY_BUDGET
        assert c.omps == DEFAULT_OMPS_PER_NODE
        assert c.ompthreads == DEFAULT_OMPTHREADS

        assert c.debug is False
        assert c.dry_run is False
        assert c.verbose == 0
        assert c.test_root is None
        assert c.requested_workers is None
        assert c.legate_dir is None

        assert c.extra_args == []
        assert c.root_dir == PurePath(m.__file__).parents[2]
        assert len(c.test_files) > 0
        assert any("examples" in str(x) for x in c.test_files)
        assert any("integration" in str(x) for x in c.test_files)
        assert all("unit" not in str(x) for x in c.test_files)
        assert c.legate_path == "legate"

    @pytest.mark.parametrize("feature", FEATURES)
    def test_env_features(
        self, monkeypatch: pytest.MonkeyPatch, feature: str
    ) -> None:
        monkeypatch.setenv(f"USE_{feature.upper()}", "1")

        # test default config
        c = m.Config([])
        assert set(c.features) == {feature}

        # also test with a --use value provided
        c = m.Config(["test.py", "--use", "cuda"])
        assert set(c.features) == {"cuda"}

    @pytest.mark.parametrize("feature", FEATURES)
    def test_cmd_features(self, feature: str) -> None:

        # test a single value
        c = m.Config(["test.py", "--use", feature])
        assert set(c.features) == {feature}

        # also test with multiple / duplication
        c = m.Config(["test.py", "--use", f"cpus,{feature}"])
        assert set(c.features) == {"cpus", feature}

    def test_unit(self) -> None:
        c = m.Config(["test.py", "--unit"])
        assert len(c.test_files) > 0
        assert any("examples" in str(x) for x in c.test_files)
        assert any("integration" in str(x) for x in c.test_files)
        assert any("unit" in str(x) for x in c.test_files)

    def test_files(self) -> None:
        c = m.Config(["test.py", "--files", "a", "b", "c"])
        assert c.files == ["a", "b", "c"]

    @pytest.mark.parametrize(
        "opt", ("cpus", "gpus", "gpu-delay", "fbmem", "omps", "ompthreads")
    )
    def test_feature_options(self, opt: str) -> None:
        c = m.Config(["test.py", f"--{opt}", "1234"])
        assert getattr(c, opt.replace("-", "_")) == 1234

    @pytest.mark.parametrize("value", PIN_OPTIONS)
    def test_cpu_pin(self, value: PinOptionsType) -> None:
        c = m.Config(["test.py", "--cpu-pin", value])
        assert c.cpu_pin == value

    def test_workers(self) -> None:
        c = m.Config(["test.py", "-j", "1234"])
        assert c.requested_workers == 1234

    def test_debug(self) -> None:
        c = m.Config(["test.py", "--debug"])
        assert c.debug is True

    def test_dry_run(self) -> None:
        c = m.Config(["test.py", "--dry-run"])
        assert c.dry_run is True

    @pytest.mark.parametrize("arg", ("-v", "--verbose"))
    def test_verbose1(self, arg: str) -> None:
        c = m.Config(["test.py", arg])
        assert c.verbose == 1

    def test_verbose2(self) -> None:
        c = m.Config(["test.py", "-vv"])
        assert c.verbose == 2

    @pytest.mark.parametrize("arg", ("-C", "--directory"))
    def test_test_root(self, arg: str) -> None:
        c = m.Config(["test.py", arg, "some/path"])
        assert c.test_root == "some/path"

    def test_legate_dir(self) -> None:
        c = m.Config([])
        assert c.legate_dir is None
        assert c.legate_path == "legate"
        assert c._legate_source == "install"

    def test_cmd_legate_dir_good(self) -> None:
        legate_dir = Path("/usr/local")
        c = m.Config(["test.py", "--legate", str(legate_dir)])
        assert c.legate_dir == legate_dir
        assert c.legate_path == str(legate_dir / "bin" / "legate")
        assert c._legate_source == "cmd"

    def test_env_legate_dir_good(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        legate_dir = Path("/usr/local")
        monkeypatch.setenv("LEGATE_DIR", str(legate_dir))
        c = m.Config([])
        assert c.legate_dir == legate_dir
        assert c.legate_path == str(legate_dir / "bin" / "legate")
        assert c._legate_source == "env"

    def test_extra_args(self) -> None:
        extra = ["-foo", "--bar", "--baz", "10"]
        c = m.Config(["test.py"] + extra)
        assert c.extra_args == extra

        # also test with --files since that option collects arguments
        c = m.Config(["test.py", "--files", "a", "b"] + extra)
        assert c.extra_args == extra
        c = m.Config(["test.py"] + extra + ["--files", "a", "b"])
        assert c.extra_args == extra
