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
"""Provide a basic logger that can scrub ANSI color codes.

"""
from __future__ import annotations

import re

# ref: https://stackoverflow.com/a/14693789
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class Log:
    def __init__(self) -> None:
        self._record: list[str] = []

    def __call__(self, *lines: str) -> tuple[int, int]:
        return self.record(*lines)

    def record(self, *lines: str) -> tuple[int, int]:
        if len(lines) == 1 and "\n" in lines[0]:
            lines = tuple(lines[0].split("\n"))

        start = len(self._record)
        for line in lines:
            self._record.append(line)
            print(line, flush=True)
        return (start, len(self._record))

    def clear(self) -> None:
        self._record = []

    def dump(
        self,
        *,
        start: int = 0,
        end: int | None = None,
        filter_ansi: bool = True,
    ) -> str:
        lines = self._record[start:end]

        if filter_ansi:
            full_text = _ANSI_ESCAPE.sub("", "\n".join(lines))
        else:
            full_text = "\n".join(lines)

        return full_text

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(self._record)


LOG = Log()
