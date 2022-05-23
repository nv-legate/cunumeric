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
from __future__ import annotations

import json
import os
from argparse import Namespace
from glob import glob
from os.path import exists, join
from pathlib import PurePath

from . import FEATURES, SKIPPED_EXAMPLES
from .args import parser


class Config:

    DEFAULT_CPUS = 4
    DEFAULT_GPUS = 1

    def __init__(self, argv: list[str]) -> None:
        args, self._extra_args = parser.parse_known_args(argv[1:])

        # which tests to run
        self.examples = True
        self.integration = True
        self.unit = args.unit
        self.files = args.files

        # feature configuration
        self.features = self._compute_features(args)

        # feature options for integration tests
        self.cpus = args.cpus
        self.gpus = args.gpus
        self.omps = args.omps
        self.ompthreads = args.ompthreads

        # test run configuration
        self.debug = args.debug
        self.dry_run = args.dry_run
        self.verbose = args.verbose
        self.workers = 1 if args.verbose else args.workers
        self.legate_dir = self._compute_legate_dir(args)

    @property
    def extra_args(self) -> list[str]:
        return self._extra_args

    @property
    def root_dir(self) -> PurePath:
        return PurePath(__file__).parents[2]

    @property
    def test_files(self) -> tuple[str, ...]:
        if self.files:
            return self.files

        files = []

        if self.examples:
            examples = (
                path
                for path in glob("examples/*.py")
                if path not in SKIPPED_EXAMPLES
            )
            files.extend(sorted(examples))

        if self.integration:
            files.extend(sorted(glob("tests/integration/*.py")))

        if self.unit:
            files.extend(sorted(glob("tests/unit/**/*.py", recursive=True)))

        return tuple(files)

    @property
    def legate_path(self) -> str:
        return join(self.legate_dir, "bin", "legate")

    def _compute_features(self, args: Namespace) -> tuple[str, ...]:
        features = set(args.features or [])
        for feature in FEATURES:
            if os.environ.get(f"USE_{feature.upper()}", None) == "1":
                features.add(feature)
        return tuple(features)

    def _compute_legate_dir(self, args: Namespace) -> str:
        if args.legate_dir:
            legate_dir = args.legate_dir

        elif "LEGATE_DIR" in os.environ:
            legate_dir = os.environ["LEGATE_DIR"]

        # TODO: This will need to change when cmake work is merged
        try:
            config_path = join(self.root_dir, ".legate.core.json")
            with open(config_path, "r") as f:
                legate_dir = json.load(f)
        except IOError:
            legate_dir = None

        if legate_dir is None or not exists(legate_dir):
            raise RuntimeError(
                "You need to provide a Legate installation directory "
                "using --legate, LEGATE_DIR or config file"
            )

        return legate_dir
