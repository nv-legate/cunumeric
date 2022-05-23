# Copyright AS2022 NVIDIA Corporation
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
from typing import Iterable

from typing_extensions import TypeAlias

Details: TypeAlias = Iterable[str]

DEFAULT_WIDTH = 60

try:
    import colorama  # type: ignore[import]

    def bright(text: str) -> str:
        return f"{colorama.Style.BRIGHT}{text}{colorama.Style.RESET_ALL}"

    def dim(text: str) -> str:
        return f"{colorama.Style.DIM}{text}{colorama.Style.RESET_ALL}"

    def white(text: str) -> str:
        return f"{colorama.Fore.WHITE}{text}{colorama.Style.RESET_ALL}"

    def cyan(text: str) -> str:
        return f"{colorama.Fore.CYAN}{text}{colorama.Style.RESET_ALL}"

    def red(text: str) -> str:
        return f"{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}"

    def green(text: str) -> str:
        return f"{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}"

    def yellow(text: str) -> str:
        return f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}"

    if sys.platform == "win32":
        colorama.init()

except ImportError:

    def _text(text: str) -> str:
        return text

    bright = dim = white = cyan = red = green = yellow = _text


def _format_details(
    details: Iterable[str] | None = None, pre: str = "   "
) -> str:
    if details:
        return f"{pre}" + f"\n{pre}".join(f"{line}" for line in details)
    return ""


def banner(
    heading: str,
    *,
    width: int = DEFAULT_WIDTH,
    details: Iterable[str] | None = None,
) -> str:
    pre = "### "
    divider = "#" * width
    if not details:
        return f"\n{divider}\n{pre}{heading}\n{divider}"
    return f"""
{divider}
{pre}
{pre}{heading}
{pre}
{_format_details(details, pre)}
{pre}
{divider}"""


def failed(msg: str, *, details: Details | None = None) -> str:
    if details:
        return f"{bright(red('[FAIL]'))} {msg}\n{_format_details(details)}"
    return f"{bright(red('[FAIL]'))} {msg}"


def passed(msg: str, *, details: Details | None = None) -> str:
    if details:
        return f"{bright(green('[PASS]'))} {msg}\n{_format_details(details)}"
    return f"{bright(green('[PASS]'))} {msg}"


def shell(cmd: str) -> str:
    return dim(white(f"+{cmd}"))


def skipped(msg: str) -> str:
    return f"{cyan('[SKIP]')} {msg}"


def bottom_line(name: str, total: int, passed: int) -> str:
    summary = (
        f"{name}: Passed {passed} of {total} tests ({passed/total*100:0.1f}%)"
    )
    if passed == total:
        return bright(green(f"{summary: >{DEFAULT_WIDTH}}"))
    return bright(red(f"{summary: >{DEFAULT_WIDTH}}"))
