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

import numpy as np
import pytest

import cunumeric as num

DIM_CASE = (5, 5)


def _print_result(test_result, print_msg, err_arr):
    assert test_result, (
        f"Failed, {print_msg}\n"
        f"Attr, {err_arr[0]}\n"
        f"numpy result: {err_arr[1]}\n"
        f"cunumeric_result: {err_arr[2]}\n"
        f"cunumeric and numpy shows"
        f" different result\n"
    )
    print(f"Passed, {print_msg}")


class Test_flags:
    arr_np = np.zeros(shape=DIM_CASE)
    arr_num = num.zeros(shape=DIM_CASE)
    attrs = [
        "c_contiguous",
        "f_contiguous",
        "owndata",
        "writeable",
        "aligned",
        "writebackifcopy",
        "behaved",
        "carray",
        "farray",
        "fnc",
        "forc",
    ]

    @pytest.mark.parametrize("view_test", [False, True], ids=bool)
    def test_default_flags_attr(self, view_test):
        b = self.arr_np.flags if view_test else self.arr_np.view().flags
        c = self.arr_num.flags if view_test else self.arr_num.view().flags
        is_equal = True
        err_arr = None
        # test default values in `ndarrray.flags`
        for attr in self.attrs:
            if getattr(b, attr) != getattr(c, attr):
                is_equal = False
                err_arr = [attr, getattr(b, attr), getattr(c, attr)]
                break

        _print_result(is_equal, "np.ndarray.flags", err_arr)

    @pytest.mark.parametrize("view_test", [False, True], ids=bool)
    def test_default_flags_element(self, view_test):
        b = self.arr_np.flags if view_test else self.arr_np.view().flags
        c = self.arr_num.flags if view_test else self.arr_num.view().flags
        is_equal = True
        err_arr = None

        short = ["C", "F", "O", "W", "A", "X", "B", "CA", "FA"]
        # test default values in `ndarrray.flags`
        for attr in self.attrs:
            attr_upper = attr.upper()
            if b[attr_upper] != c[attr_upper]:
                is_equal = False
                err_arr = [attr, b[attr_upper], c[attr_upper]]
                break

        for idx, attr in enumerate(short):
            if b[attr] != c[attr]:
                is_equal = False
                err_arr = [attr, b[attr], c[attr]]
                break
            if b[attr] is not b[self.attrs[idx].upper()]:
                is_equal = False
                err_arr = [
                    (attr, self.attrs[idx].upper()),
                    b[attr],
                    self.attrs[idx].upper(),
                ]
                break

        _print_result(is_equal, "np.ndarray.flags", err_arr)

    @pytest.mark.parametrize("view_test", [False, True], ids=bool)
    def test_setflags(self, view_test):
        b = self.arr_np if view_test else self.arr_np.view()
        c = self.arr_num if view_test else self.arr_num.view()
        is_equal = True
        err_arr = None
        for attr in self.attrs[3:6]:
            attr = attr.upper()
            # alter flags
            error_b = False
            error_c = False
            try:
                b.flags[attr] = not b.flags[attr]
            except ValueError:
                error_b = True
            try:
                c.flags[attr] = not c.flags[attr]
            except ValueError:
                error_c = True
            if attr == "WRITEBACKIFCOPY":
                is_equal = error_b and error_c
            else:
                is_equal = b.flags[attr] == c.flags[attr]
            err_arr = [("flags", attr), b.flags[attr], c.flags[attr]]
            if not is_equal:
                break
        _print_result(is_equal, "np.ndarray.flags", err_arr)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
