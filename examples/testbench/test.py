#!/usr/bin/env python

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

from __future__ import print_function

import argparse
import datetime

#import pdb
#pdb.set_trace()
import legate.numpy as np

#import numpy as np
#import numpy as np
N=500000
M=2
b1 = np.zeros(N)
b2 = np.zeros(N)
b3 = np.zeros(N)
b4 = np.zeros(N)
b5 = np.zeros(N)

b1.fill(2)
b2.fill(3)
b3.fill(4)

#dummy ops
# dumb hack so the loop starts at nOps=10
# so the window/pipe is empty
b3.fill(4)
b3.fill(4)
#gg=b1+b2+b3
#print(gg)
#for i in range(10000):
for i in range(1000):
    b1=b1+b2+b3+b4+b5+b1+b2+b3+b4+b5+b1
    #b3+=b3
    #b3=b1+b2
    
#print(b3)
"""
def shor_form(a):
    a+=a
    return a
def long_form(a):
    a=a+a
    return a
import ast,inspect
func_src = inspect.getsource(long_form)
entry_node = ast.parse(func_src)
print(ast.dump(entry_node))

func_src = inspect.getsource(shor_form)
entry_node = ast.parse(func_src)
print(ast.dump(entry_node))




#a=a+a
#a=a+a
#b=mul_add(b, b1,b2)
#print(sum(g))


long_form(b,b1,b2)
#c = np.sum(a, axis=0)
c = np.dot(a,b)
#c = np.dot(a,b)
#c = np.dot(a,b)

#sink?
print(sum(c))



n1 = 300
n2= 20
a = npo.zeros((n1,n2))
b = npo.zeros((N))
c= npo.zeros((N))
print("s",sum(long_form(a,b,c)))
a+=a
M=800000
d0 = np.zeros(M)
d0.fill(1)
d1 = np.zeros(M)
d2 = np.zeros(M)
print(sum(e))
"""
