# Copyright 2023 NVIDIA Corporation
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
import numpy as np
from legate.core import LEGATE_MAX_DIM
from utils.generators import mk_seq_array

def my_func(a, b):
    a = a * 2 + b
    return a

#Capital letters and numbers in the signature 
def my_func2(A0, B0):
    A0 = A0 * 2 + B0
    C0=A0*2
    return A0,C0

def test_vectorize():
    #2 arrays
    func = num.vectorize(my_func)
    a = num.arange(5)
    b = num.ones((5,))
    a = func(a, b)
    assert(np.array_equal(a, [1,3,5,7,9]))

    #array and scalar
    func = num.vectorize(my_func)
    a= num.arange(5)
    b=2
    a = func(a,b)
    assert(np.array_equal(a, [2,4,6,8,10]))
   
    #2 scalars
    func = num.vectorize(my_func)
    a=3
    b=2
    a = func(a,b)
    assert(a ==8)

def empty_func():
    print("within empty function")

def print_func(a,b):
    print ("I am pringing input arguments", a, b)

def test_empty_functions():
    #empty function
    func = num.vectorize(empty_func)
    func()

    func2 = num.vectorize(print_func)
    print_func(1,2)

    print_func(np.array([1,2,3]), 2)


def test_vectorize_over_slices():
    #reuse the same vectorize object on
    #different slices
    func_num = num.vectorize(my_func)
    func_np = np.vectorize(my_func)

    a=np.array([[1,2,3],[4,5,6],[7,8,9]])
    b=np.array([[10,11,12],[13,14,15],[16,17,18]])
    a_num=num.array(a)
    b_num = num.array(b)
    a[:2] = func_np(a[:2],b[:2])
    a_num[:2] = func_num(a_num[:2],b_num[:2])
    assert np.array_equal(a, a_num)


    a=np.arange(100).reshape((25,4))
    a_num= num.array(a)
    b=a*10
    b_num=a_num*10
    a=func_np(a,b)
    a_num=func_num(a_num, b_num)
    assert np.array_equal(a, a_num)

    #reusing the same function for different inputs
    a[:,2]=func_np(a[:, 2], b[:,2])
    a_num[:,2] =func_num(a_num[:,2],b_num[:,2])
    assert np.array_equal(a, a_num)

    #reusing the same function for different inputs
    a[5:10,2]=func_np(a[5:10, 2], b[1:6,2])
    a_num[5:10,2]=func_num(a_num[5:10,2],b_num[1:6,2])
    assert np.array_equal(a, a_num)

    #reusing the same function for different inputs
    a[15:20]=func_np(a[15:20], b[15:20])
    a_num[15:20]=func_num(a_num[15:20],b_num[15:20])
    assert np.array_equal(a, a_num)

    # reusing the same function for different inputs
    a=np.arange(1000).reshape((25,10,4))
    a_num= num.array(a)
    a[:, 2, :] = func_np(a[:, 2, :],2)
    a_num[:, 2, :]=func_num(a_num[:, 2, :],2)
    assert np.array_equal(a, a_num)

def test_multiple_outputs():
    #checking signature with capital letters and numbers
    # + checking multiple outputs
    a=np.arange(100).reshape((25,4))
    a_num= num.array(a)
    b=a*10
    b_num=a_num*10
    func_np = np.vectorize(my_func2)
    func_num=num.vectorize(my_func2)
    a,c=func_np(a,b)
    a_num,c_num = func_num(a_num, b_num)
    assert np.array_equal(a, a_num)
    assert np.array_equal(c, c_num)

def test_different_types():
    #checking the case when input and output types are different
    a=np.arange(100, dtype = int).reshape((25,4))
    a_num= num.array(a)
    b=a*10
    b_num=a_num*10
    func_np = np.vectorize(my_func, otypes=(float,))
    func_num=num.vectorize(my_func, otypes=(float,))
    a=func_np(a,b)
    a_num=func_num(a_num, b_num)
    assert np.array_equal(a, a_num)

    #another test for different types
    a=np.arange(100, dtype = float).reshape((25,4))
    a_num= num.array(a)
    b=a*10
    b_num=a_num*10
    func_np = np.vectorize(my_func2, otypes = (int, int,))
    func_num=num.vectorize(my_func2, otypes = (int, int, ))
    a,c=func_np(a,b)
    a_num,c_num = func_num(a_num, b_num)
    assert np.array_equal(a, a_num)
    assert np.array_equal(c, c_num)


def test_cache():
    a=np.arange(100).reshape((25,4))
    a_num= num.array(a)
    b=a*10
    b_num=a_num*10
    func_np = np.vectorize(my_func2, cache = True)
    func_num=num.vectorize(my_func2, cache = True)
    for i in range (10):
        a=a*2
        b=b*3
        a_num=a_num*2
        b_num=b_num*3
        a,c=func_np(a,b)
        a_num,c_num = func_num(a_num, b_num)
        assert np.array_equal(a, a_num)
        assert np.array_equal(c, c_num)

#checking caching on different shapes of arrays:
func_np2 = np.vectorize(my_func2, cache = True)
func_num2=num.vectorize(my_func2, cache = True)

@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_nd_vectorize(ndim):
    a_shape = tuple(np.random.randint(1, 9) for _ in range(ndim))
    a = mk_seq_array(np, a_shape)
    a_num = mk_seq_array(num, a_shape)
    b=a*2
    b_num=a_num*2
    a,c=func_np2(a,b)
    a_num,c_num = func_num2(a_num, b_num)
    assert np.array_equal(a, a_num)
    assert np.array_equal(c, c_num)        

if __name__ == "__main__":
    import sys

    np.random.seed(12345)
    sys.exit(pytest.main(sys.argv))
