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
"""Provide an argparse ArgumentParser for the test runner.

"""
from __future__ import annotations

from argparse import Action, ArgumentParser, Namespace
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    TypeVar,
    Union,
)

from typing_extensions import TypeAlias

from . import (
    DEFAULT_CPUS_PER_NODE,
    DEFAULT_GPU_DELAY,
    DEFAULT_GPU_MEMORY_BUDGET,
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_OMPS_PER_NODE,
    DEFAULT_OMPTHREADS,
    FEATURES,
)

T = TypeVar("T")

PinOptionsType: TypeAlias = Union[
    Literal["partial"],
    Literal["none"],
    Literal["strict"],
]

PIN_OPTIONS: tuple[PinOptionsType, ...] = (
    "partial",
    "none",
    "strict",
)


class MultipleChoices(Generic[T]):
    """A container that reports True for any item or subset inclusion.

    Parameters
    ----------
    choices: Iterable[T]
        The values to populate the containter.

    Examples
    --------

    >>> choices = MultipleChoices(["a", "b", "c"])

    >>> "a" in choices
    True

    >>> ("b", "c") in choices
    True

    """

    def __init__(self, choices: Iterable[T]) -> None:
        self.choices = set(choices)

    def __contains__(self, x: Union[T, Iterable[T]]) -> bool:
        if isinstance(x, (list, tuple)):
            return set(x).issubset(self.choices)
        return x in self.choices

    def __iter__(self) -> Iterator[T]:
        return self.choices.__iter__()


class ExtendAction(Action):
    """A custom argparse action to collect multiple values into a list."""

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


#: The argument parser for test.py
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
    # argpase evidently only expects string returns from the type converter
    # here, but returning a list of strings seems to work in practice
    type=lambda s: s.split(","),  # type: ignore[return-value, arg-type]
    help="Test Legate with features (also via USE_*)",
)


selection = parser.add_argument_group("Test file selection")


selection.add_argument(
    "--files",
    nargs="+",
    default=None,
    help="Explicit list of test files to run",
)


selection.add_argument(
    "--unit",
    dest="unit",
    action="store_true",
    default=False,
    help="Include unit tests",
)


feature_opts = parser.add_argument_group("Feature stage configuration options")


feature_opts.add_argument(
    "--cpus",
    dest="cpus",
    type=int,
    default=DEFAULT_CPUS_PER_NODE,
    help="Number of CPUs per node to use",
)


feature_opts.add_argument(
    "--gpus",
    dest="gpus",
    type=int,
    default=DEFAULT_GPUS_PER_NODE,
    help="Number of GPUs per node to use",
)


feature_opts.add_argument(
    "--omps",
    dest="omps",
    type=int,
    default=DEFAULT_OMPS_PER_NODE,
    help="Number OpenMP processors per node to use",
)


feature_opts.add_argument(
    "--utility",
    dest="utility",
    type=int,
    default=1,
    help="Number of of utility CPUs to reserve for runtime services",
)


feature_opts.add_argument(
    "--cpu-pin",
    dest="cpu_pin",
    choices=PIN_OPTIONS,
    default="partial",
    help="CPU pinning behavior on platforms that support CPU pinning",
)

feature_opts.add_argument(
    "--gpu-delay",
    dest="gpu_delay",
    type=int,
    default=DEFAULT_GPU_DELAY,
    help="Delay to introduce between GPU tests (ms)",
)


feature_opts.add_argument(
    "--fbmem",
    dest="fbmem",
    type=int,
    default=DEFAULT_GPU_MEMORY_BUDGET,
    help="GPU framebuffer memory (MB)",
)


feature_opts.add_argument(
    "--ompthreads",
    dest="ompthreads",
    metavar="THREADS",
    type=int,
    default=DEFAULT_OMPTHREADS,
    help="Number of threads per OpenMP processor",
)


test_opts = parser.add_argument_group("Test run configuration options")


test_opts.add_argument(
    "--legate",
    dest="legate_dir",
    metavar="LEGATE_DIR",
    action="store",
    default=None,
    required=False,
    help="Path to Legate installation directory",
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
    "-j",
    "--workers",
    dest="workers",
    type=int,
    default=None,
    help="Number of parallel workers for testing",
)


test_opts.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action="count",
    default=0,
    help="Display verbose output. Use -vv for even more output (test stdout)",
)


test_opts.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
    help="Print the test plan but don't run anything",
)


test_opts.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Print out the commands that are to be executed",
)
