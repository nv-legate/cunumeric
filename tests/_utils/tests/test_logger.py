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

from .. import logger as m

TEST_LINES = (
    "line 1",
    "\x1b[31mfoo\x1b[0m",  # ui.red("foo")
    "bar",
    "last line",
)


class TestLogger:
    def test_init(self) -> None:
        log = m.Log()
        assert log.lines == ()
        assert log.dump() == ""

    def test_record_lines(self) -> None:
        log = m.Log()
        log.record(*TEST_LINES)
        assert log.lines == TEST_LINES
        assert log.dump(filter_ansi=False) == "\n".join(TEST_LINES)

    def test_record_line_with_newlines(self) -> None:
        log = m.Log()
        log.record("\n".join(TEST_LINES))
        assert log.lines == TEST_LINES
        assert log.dump(filter_ansi=False) == "\n".join(TEST_LINES)

    def test_call(self) -> None:
        log = m.Log()
        log(*TEST_LINES)
        assert log.lines == TEST_LINES
        assert log.dump() == "line 1\nfoo\nbar\nlast line"

    def test_dump_filter(self) -> None:
        log = m.Log()
        log.record(*TEST_LINES)
        assert log.lines == TEST_LINES
        assert log.dump() == "line 1\nfoo\nbar\nlast line"

    def test_dump_index(self) -> None:
        log = m.Log()
        log.record(*TEST_LINES)
        assert log.dump(start=1, end=3) == "foo\nbar"

    def test_clear(self) -> None:
        log = m.Log()
        log.record(*TEST_LINES)
        assert len(log.lines) > 0
        log.clear()
        assert len(log.lines) == 0


def test_LOG() -> None:
    assert isinstance(m.LOG, m.Log)
