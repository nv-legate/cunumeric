# Copyright 2021 NVIDIA Corporation
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

from __future__ import absolute_import

from universal_functions_tests import (
    absolute,
    add,
    arccos,
    arcsin,
    arctan,
    ceil,
    clip,
    convert,
    cos,
    divide,
    equal,
    exp,
    exp2,
    floor,
    floor_divide,
    greater,
    greater_equal,
    invert,
    isinf,
    isnan,
    less,
    less_equal,
    log,
    log10,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    multiply,
    negative,
    not_equal,
    power,
    rint,
    sign,
    sin,
    sqrt,
    subtract,
    tan,
    tanh,
    true_divide,
)


def test():
    absolute.test()
    add.test()
    arccos.test()
    arcsin.test()
    arctan.test()
    ceil.test()
    clip.test()
    convert.test()
    cos.test()
    divide.test()
    equal.test()
    exp.test()
    exp2.test()
    floor.test()
    floor_divide.test()
    greater.test()
    greater_equal.test()
    invert.test()
    isinf.test()
    isnan.test()
    less.test()
    less_equal.test()
    log.test()
    log10.test()
    logical_and.test()
    logical_not.test()
    logical_or.test()
    logical_xor.test()
    multiply.test()
    negative.test()
    not_equal.test()
    power.test()
    rint.test()
    sign.test()
    sin.test()
    sqrt.test()
    subtract.test()
    tan.test()
    tanh.test()
    true_divide.test()


if __name__ == "__main__":
    test()
