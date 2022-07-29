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
"""Helpler functions for simple text UI output.

The color functions in this module require ``colorama`` to be installed in
order to generate color output. If ``colorama`` is not available, plain
text output (i.e. without ANSI color codes) will generated.

"""
from __future__ import annotations

import sys
from typing import Iterable

from typing_extensions import TypeAlias

from . import UI_WIDTH

Details: TypeAlias = Iterable[str]


def _text(text: str) -> str:
    return text


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
    char: str = "#",
    width: int = UI_WIDTH,
    details: Iterable[str] | None = None,
) -> str:
    """Generate a title banner, with optional details included.

    Parameters
    ----------
    heading : str
        Text to use for the title

    char : str, optional
        A character to use to frame the banner. (default: "#")

    width : int, optional
        How wide to draw the banner. (Note: user-supplied heading or
        details willnot be truncated if they exceed this width)

    details : Iterable[str], optional
        A list of lines to diplay inside the banner area below the heading

    """
    pre = f"{char*3} "
    divider = char * width
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
    """Report a failed test result with a bright red [FAIL].

    Parameters
    ----------
    msg : str
        Text to display after [FAIL]

    details : Iterable[str], optional
        A sequenece of text lines to diplay below the ``msg`` line

    """
    if details:
        return f"{bright(red('[FAIL]'))} {msg}\n{_format_details(details)}"
    return f"{bright(red('[FAIL]'))} {msg}"


def passed(msg: str, *, details: Details | None = None) -> str:
    """Report a passed test result with a bright green [PASS].

    Parameters
    ----------
    msg : str
        Text to display after [PASS]

    details : Iterable[str], optional
        A sequenece of text lines to diplay below the ``msg`` line

    """
    if details:
        return f"{bright(green('[PASS]'))} {msg}\n{_format_details(details)}"
    return f"{bright(green('[PASS]'))} {msg}"


def rule(pad: int = 4, char: str = "~") -> str:
    """Generate a horizontal rule.

    Parameters
    ----------
    pad : int, optional
        How much whitespace to precede the rule. (default: 4)

    char : str, optional
        A character to use to "draw" the rule. (default: "~")

    """
    w = UI_WIDTH - pad
    return f"{char*w: >{UI_WIDTH}}"


def shell(cmd: str, *, char: str = "+") -> str:
    """Report a shell command in a dim white color.

    Parameters
    ----------
    cmd : str
        The shell command string to display

    char : str, optional
        A character to prefix the ``cmd`` with. (default: "+")

    """
    return dim(white(f"{char}{cmd}"))


def skipped(msg: str) -> str:
    """Report a skipped test with a cyan [SKIP]

    Parameters
    ----------
    msg : str
        Text to display after [SKIP]

    """
    return f"{cyan('[SKIP]')} {msg}"


def summary(name: str, total: int, passed: int) -> str:
    """Generate a test result summary line.

    The output is bright green if all tests passed, otherwise bright red.

    Parameters
    ----------
    name : str
        A name to display in this summary line.

    total : int
        The total number of tests to report.

    passed : int
        The number of passed tests to report.

    """
    summary = (
        f"{name}: Passed {passed} of {total} tests ({passed/total*100:0.1f}%)"
    )
    color = green if passed == total else red
    return bright(color(f"{summary: >{UI_WIDTH}}"))
