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

from argparse import Action, ArgumentParser, Namespace
from typing import Any, Generic, Iterable, Iterator, Sequence, TypeVar, Union

from . import FEATURES

T = TypeVar("T")


class MultipleChoices(Generic[T]):
    def __init__(self, choices: Iterable[T]) -> None:
        self.choices = set(choices)

    def __contains__(self, x: Union[T, Iterable[T]]) -> bool:
        if isinstance(x, list):
            return set(x).issubset(self.choices)
        return x in self.choices

    def __iter__(self) -> Iterator[T]:
        return self.choices.__iter__()


class ExtendAction(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Union[str, Sequence[Any], None],
        option_string: Union[str, None] = None,
    ) -> None:
        items = getattr(namespace, self.dest, None) or []
        if isinstance(values, list):
            items.extend(values)
        else:
            items.append(values)
        setattr(namespace, self.dest, items)


parser = ArgumentParser(
    description="Run the Cunumeric test suite",
    epilog="Any extra arguments will be forwarded to the Legate script",
)


stages = parser.add_argument_group("Feature stage selection")


stages.add_argument(
    "--use",
    dest="features",
    action=ExtendAction,
    choices=MultipleChoices(sorted(FEATURES)),
    type=lambda s: s.split(","),  # type: ignore[return-value, arg-type]
    help="Test Legate with features (also via USE_*).",
)


selection = parser.add_argument_group("Test file selection")


selection.add_argument(
    "files",
    nargs="*",
    default=None,
    help="Explicit list of test files to run.",
)


selection.add_argument(
    "--unit",
    dest="unit",
    action="store_true",
    default=False,
    help="Include unit tests.",
)


feature_opts = parser.add_argument_group("Feature stage configuration options")


feature_opts.add_argument(
    "--cpus",
    type=int,
    default=4,
    dest="cpus",
    help="Number of CPUs per node to use.",
)


feature_opts.add_argument(
    "--gpus",
    type=int,
    default=1,
    dest="gpus",
    help="Number of GPUs per node to use.",
)


feature_opts.add_argument(
    "--omps",
    type=int,
    default=1,
    dest="omps",
    help="Number OpenMP processors per node to use.",
)


feature_opts.add_argument(
    "--ompthreads",
    type=int,
    default=4,
    dest="ompthreads",
    help="Number of threads per OpenMP processor.",
)


test_opts = parser.add_argument_group("Test run configuration options")


test_opts.add_argument(
    "--legate",
    dest="legate_dir",
    metavar="LEGATE_DIR",
    action="store",
    help="Path to Legate installation directory.",
)


test_opts.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action="store_true",
    help="Print more debugging information.",
)


test_opts.add_argument(
    "-j",
    type=int,
    default=None,
    dest="workers",
    help="Number of parallel workers for testing",
)

test_opts.add_argument(
    "-C",
    "--directory",
    dest="test_root",
    metavar="DIR",
    action="store",
    default=None,
    required=False,
    help="Root directory containing the tests subdirectory",
)


test_opts.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
    help="Print the test plan but don't run anything.",
)


test_opts.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Print out the commands that are to be executed.",
)
