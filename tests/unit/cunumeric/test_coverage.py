# Copyright 2021-2022 NVIDIA Corporation
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

from types import ModuleType

import cunumeric.coverage as m  # module under test
import pytest
from mock import MagicMock, patch

import cunumeric


def test_FALLBACK_WARNING() -> None:
    assert m.FALLBACK_WARNING.format(name="foo") == (
        "cuNumeric has not implemented foo "
        + "and is falling back to canonical numpy. "
        + "You may notice significantly decreased performance "
        + "for this function call."
    )


def test_MOD_INTERNAL() -> None:
    assert m.MOD_INTERNAL == {"__dir__", "__getattr__"}


def test_NDARRAY_INTERNAL() -> None:
    assert m.NDARRAY_INTERNAL == {
        "__array_finalize__",
        "__array_function__",
        "__array_interface__",
        "__array_prepare__",
        "__array_priority__",
        "__array_struct__",
        "__array_ufunc__",
        "__array_wrap__",
    }


class Test_filter_namespace:
    def test_empty(self) -> None:
        orig = dict()
        ns = dict(orig)

        result = m.filter_namespace(ns)
        assert orig == ns
        assert result == ns
        assert result is not ns

        result = m.filter_namespace(ns, omit_names=("foo",))
        assert orig == ns
        assert result == ns
        assert result is not ns

        result = m.filter_namespace(ns, omit_types=(int, str))
        assert orig == ns
        assert result == ns
        assert result is not ns

    def test_no_filters(self) -> None:
        orig = dict(foo=10)
        ns = dict(orig)

        result = m.filter_namespace(ns)
        assert orig == ns
        assert result == ns
        assert result is not ns

    def test_name_filters(self) -> None:
        orig = dict(foo=10, bar=20)
        ns = dict(orig)

        result = m.filter_namespace(ns, omit_names=("foo",))
        assert orig == ns
        assert result == dict(bar=20)

        result = m.filter_namespace(ns, omit_names=("foo", "bar"))
        assert orig == ns
        assert result == dict()

        result = m.filter_namespace(ns, omit_names=("foo", "baz"))
        assert orig == ns
        assert result == dict(bar=20)

    def test_type_filters(self) -> None:
        orig = dict(foo=10, bar="abc")
        ns = dict(orig)

        result = m.filter_namespace(ns, omit_types=(int,))
        assert orig == ns
        assert result == dict(bar="abc")

        result = m.filter_namespace(ns, omit_types=(int, str))
        assert orig == ns
        assert result == dict()

        result = m.filter_namespace(ns, omit_types=(int, float))
        assert orig == ns
        assert result == dict(bar="abc")


def _test_func(a: int, b: int) -> int:
    """docstring"""
    return a + b


class _Test_ufunc(cunumeric._ufunc.ufunc):
    """docstring"""

    def __call__(self, a: int, b: int) -> int:
        return a + b


_test_ufunc = _Test_ufunc()


class Test_implemented:
    @patch("cunumeric.runtime.record_api_call")
    def test_reporting_True_func(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.implemented(_test_func, "foo", "_test_func")

        assert wrapped.__doc__ == _test_func.__doc__
        assert wrapped.__wrapped__ is _test_func

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_called_once()
        assert mock_record_api_call.call_args[0] == ()
        assert mock_record_api_call.call_args[1]["name"] == "foo._test_func"
        assert mock_record_api_call.call_args[1]["implemented"]
        filename, lineno = mock_record_api_call.call_args[1]["location"].split(
            ":"
        )
        assert filename == __file__
        assert int(lineno)

    @patch("cunumeric.runtime.record_api_call")
    def test_reporting_False_func(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.implemented(
            _test_func, "foo", "_test_func", reporting=False
        )

        assert wrapped.__doc__ == _test_func.__doc__
        assert wrapped.__wrapped__ is _test_func

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_not_called()

    @patch("cunumeric.runtime.record_api_call")
    def test_reporting_True_ufunc(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.implemented(_test_ufunc, "foo", "_test_ufunc")

        assert wrapped.__doc__ == _test_ufunc.__doc__
        assert wrapped.__wrapped__ is _test_ufunc

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_called_once()
        assert mock_record_api_call.call_args[0] == ()
        assert mock_record_api_call.call_args[1]["name"] == "foo._test_ufunc"
        assert mock_record_api_call.call_args[1]["implemented"]
        filename, lineno = mock_record_api_call.call_args[1]["location"].split(
            ":"
        )
        assert filename == __file__
        assert int(lineno)

    @patch("cunumeric.runtime.record_api_call")
    def test_reporting_False_ufunc(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.implemented(
            _test_ufunc, "foo", "_test_func", reporting=False
        )

        assert wrapped.__doc__ == _test_ufunc.__doc__
        assert wrapped.__wrapped__ is _test_ufunc

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_not_called()


class Test_unimplemented:
    @patch("cunumeric.runtime.record_api_call")
    def test_reporting_True_func(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.unimplemented(_test_func, "foo", "_test_func")

        assert wrapped.__doc__ == _test_func.__doc__
        assert wrapped.__wrapped__ is _test_func

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_called_once()
        assert mock_record_api_call.call_args[0] == ()
        assert mock_record_api_call.call_args[1]["name"] == "foo._test_func"
        assert not mock_record_api_call.call_args[1]["implemented"]
        filename, lineno = mock_record_api_call.call_args[1]["location"].split(
            ":"
        )
        assert filename == __file__
        assert int(lineno)

    @patch("cunumeric.runtime.record_api_call")
    def test_reporting_False_func(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.unimplemented(
            _test_func, "foo", "_test_func", reporting=False
        )

        assert wrapped.__doc__ == _test_func.__doc__
        assert wrapped.__wrapped__ is _test_func

        with pytest.warns(RuntimeWarning) as record:
            assert wrapped(10, 20) == 30

        assert len(record) == 1
        assert record[0].message.args[0] == m.FALLBACK_WARNING.format(
            name="foo._test_func"
        )

        mock_record_api_call.assert_not_called()

    @patch("cunumeric.runtime.record_api_call")
    def test_reporting_True_ufunc(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.unimplemented(_test_ufunc, "foo", "_test_ufunc")

        assert wrapped.__doc__ == _test_ufunc.__doc__
        assert wrapped.__wrapped__ is _test_ufunc

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_called_once()
        assert mock_record_api_call.call_args[0] == ()
        assert mock_record_api_call.call_args[1]["name"] == "foo._test_ufunc"
        assert not mock_record_api_call.call_args[1]["implemented"]
        filename, lineno = mock_record_api_call.call_args[1]["location"].split(
            ":"
        )
        assert filename == __file__
        assert int(lineno)

    @patch("cunumeric.runtime.record_api_call")
    def test_reporting_False_ufunc(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.unimplemented(
            _test_ufunc, "foo", "_test_ufunc", reporting=False
        )

        assert wrapped.__doc__ == _test_ufunc.__doc__
        assert wrapped.__wrapped__ is _test_ufunc

        with pytest.warns(RuntimeWarning) as record:
            assert wrapped(10, 20) == 30

        assert len(record) == 1
        assert record[0].message.args[0] == m.FALLBACK_WARNING.format(
            name="foo._test_ufunc"
        )

        mock_record_api_call.assert_not_called()


_OriginMod = ModuleType("origin")
exec(
    """

import numpy as np

def __getattr__(name):
    pass

def __dir__():
    pass

attr1 = 10
attr2 = 20

def function1():
    pass

def function2():
    pass

""",
    _OriginMod.__dict__,
)

_DestCode = """
def function2():
    pass

attr2 = 30
"""


class Test_clone_module:
    @patch.object(cunumeric.runtime, "report_coverage", True)
    def test_report_coverage_True(self) -> None:
        assert cunumeric.runtime.report_coverage

        _Dest = ModuleType("dest")
        exec(_DestCode, _Dest.__dict__)

        m.clone_module(_OriginMod, _Dest.__dict__)

        for name in m.MOD_INTERNAL:
            assert name not in _Dest.__dict__

        assert "np" not in _Dest.__dict__

        assert _Dest.attr1 == 10
        assert _Dest.attr2 == 30

        assert _Dest.function1.__wrapped__ is _OriginMod.function1
        assert not _Dest.function1._cunumeric.implemented

        assert _Dest.function2.__wrapped__
        assert _Dest.function2._cunumeric.implemented

    @patch.object(cunumeric.runtime, "report_coverage", False)
    def test_report_coverage_False(self) -> None:
        assert not cunumeric.runtime.report_coverage

        _Dest = ModuleType("dest")
        exec(_DestCode, _Dest.__dict__)

        m.clone_module(_OriginMod, _Dest.__dict__)

        for name in m.MOD_INTERNAL:
            assert name not in _Dest.__dict__

        assert "np" not in _Dest.__dict__

        assert _Dest.attr1 == 10
        assert _Dest.attr2 == 30

        assert _Dest.function1.__wrapped__ is _OriginMod.function1
        assert not _Dest.function1._cunumeric.implemented

        assert _Dest.function2 is _Dest.function2


class _OriginClass:
    def __array_finalize__(self) -> None:
        pass

    def __array_function__(self) -> None:
        pass

    def __array_interface__(self) -> None:
        pass

    def __array_prepare__(self) -> None:
        pass

    def __array_priority__(self) -> None:
        pass

    def __array_struct__(self) -> None:
        pass

    def __array_ufunc__(self) -> None:
        pass

    def __array_wrap__(self) -> None:
        pass

    def method1(self) -> None:
        pass

    def method2(self) -> None:
        pass

    attr1 = 10

    attr2 = 20


class Test_clone_class:
    @patch.object(cunumeric.runtime, "report_coverage", True)
    def test_report_coverage_True(self) -> None:
        assert cunumeric.runtime.report_coverage

        @m.clone_class(_OriginClass)
        class _Dest:
            def method2(self) -> None:
                pass

            attr2 = 30

        for name in m.NDARRAY_INTERNAL:
            assert name not in _Dest.__dict__

        assert _Dest.attr1 == 10
        assert _Dest.attr2 == 30

        assert _Dest.method1.__wrapped__ is _OriginClass.method1
        assert not _Dest.method1._cunumeric.implemented

        assert _Dest.method2.__wrapped__
        assert _Dest.method2._cunumeric.implemented

    @patch.object(cunumeric.runtime, "report_coverage", False)
    def test_report_coverage_False(self) -> None:
        assert not cunumeric.runtime.report_coverage

        @m.clone_class(_OriginClass)
        class _Dest:
            def method2(self) -> None:
                pass

            attr2 = 30

        for name in m.NDARRAY_INTERNAL:
            assert name not in _Dest.__dict__

        assert _Dest.attr1 == 10
        assert _Dest.attr2 == 30

        assert _Dest.method1.__wrapped__ is _OriginClass.method1
        assert not _Dest.method1._cunumeric.implemented

        assert _Dest.method2 is _Dest.method2


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
