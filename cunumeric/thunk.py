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

from abc import ABC, abstractmethod, abstractproperty


class NumPyThunk(ABC):
    """This is the base class for NumPy computations. It has methods
    for all the kinds of computations and operations that can be done
    on cuNumeric ndarrays.

    :meta private:
    """

    def __init__(self, runtime, dtype):
        self.runtime = runtime
        self.context = runtime.legate_context
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        s = 1
        if self.ndim == 0:
            return s
        for p in self.shape:
            s *= p
        return s

    # Abstract methods

    @abstractproperty
    def storage(self):
        """Return the Legion storage primitive for this NumPy thunk"""
        ...

    @abstractmethod
    def __numpy_array__(self):
        ...

    @abstractmethod
    def imag(self):
        ...

    @abstractmethod
    def real(self):
        ...

    @abstractmethod
    def conj(self):
        ...

    @abstractmethod
    def convolve(self, v, out, mode):
        ...

    @abstractmethod
    def fft(self, out, axes, kind, direction):
        ...

    @abstractmethod
    def copy(self, rhs, deep):
        ...

    @abstractmethod
    def repeat(self, repeats, axis, scalar_repeats):
        ...

    @property
    @abstractmethod
    def scalar(self):
        ...

    @abstractmethod
    def get_scalar_array(self):
        ...

    @abstractmethod
    def get_item(self, key, view=None, dim_map=None):
        ...

    @abstractmethod
    def set_item(self, key, value):
        ...

    @abstractmethod
    def reshape(self, newshape, order):
        ...

    @abstractmethod
    def squeeze(self, axis):
        ...

    @abstractmethod
    def swapaxes(self, axis1, axis2):
        ...

    @abstractmethod
    def convert(self, rhs, warn=True):
        ...

    @abstractmethod
    def fill(self, value):
        ...

    @abstractmethod
    def transpose(self, rhs, axes):
        ...

    @abstractmethod
    def flip(self, rhs, axes):
        ...

    @abstractmethod
    def contract(
        self,
        lhs_modes,
        rhs1_thunk,
        rhs1_modes,
        rhs2_thunk,
        rhs2_modes,
        mode2extent,
    ):
        ...

    @abstractmethod
    def choose(self, *args, rhs):
        ...

    @abstractmethod
    def _diag_helper(self, rhs, offset, naxes, extract, trace):
        ...

    @abstractmethod
    def eye(self, k):
        ...

    @abstractmethod
    def arange(self, start, stop, step):
        ...

    @abstractmethod
    def tile(self, rhs, reps):
        ...

    @abstractmethod
    def trilu(self, start, stop, step):
        ...

    @abstractmethod
    def bincount(self, rhs, weights=None):
        ...

    @abstractmethod
    def nonzero(self):
        ...

    @abstractmethod
    def random_uniform(self):
        ...

    @abstractmethod
    def random_normal(self):
        ...

    @abstractmethod
    def random_integer(self, low, high):
        ...

    @abstractmethod
    def unary_op(self, op, rhs, where, args):
        ...

    @abstractmethod
    def unary_reduction(
        self,
        op,
        redop,
        rhs,
        where,
        axes,
        keepdims,
        args,
        initial,
    ):
        ...

    @abstractmethod
    def binary_op(self, op, rhs1, rhs2, where, args):
        ...

    @abstractmethod
    def binary_reduction(self, op, rhs1, rhs2, broadcast, args):
        ...

    @abstractmethod
    def where(self, op, rhs1, rhs2, rhs3):
        ...

    @abstractmethod
    def cholesky(self, src, no_tril):
        ...

    @abstractmethod
    def unique(self):
        ...

    @abstractmethod
    def create_window(self, op_code, *args):
        ...
