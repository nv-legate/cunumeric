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
from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy.typing as npt

    from .types import NdShape


class NumPyThunk(ABC):
    """This is the base class for NumPy computations. It has methods
    for all the kinds of computations and operations that can be done
    on cuNumeric ndarrays.

    :meta private:
    """

    def __init__(self, runtime, dtype) -> None:
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

    @abstractproperty
    def shape(self) -> NdShape:
        ...

    @abstractmethod
    def __numpy_array__(self) -> npt.NDArray[Any]:
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
    def convolve(self, v, out, mode) -> None:
        ...

    @abstractmethod
    def fft(self, out, axes, kind, direction):
        ...

    @abstractmethod
    def copy(self, rhs, deep) -> None:
        ...

    @abstractmethod
    def repeat(self, repeats, axis, scalar_repeats) -> NumPyThunk:
        ...

    @property
    @abstractmethod
    def scalar(self):
        ...

    @abstractmethod
    def get_scalar_array(self):
        ...

    @abstractmethod
    def get_item(self, key, view=None, dim_map=None) -> NumPyThunk:
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
    def convert(self, rhs, warn=True) -> None:
        ...

    @abstractmethod
    def fill(self, value) -> None:
        ...

    @abstractmethod
    def transpose(self, axes):
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
    ) -> None:
        ...

    @abstractmethod
    def choose(self, *args, rhs):
        ...

    @abstractmethod
    def _diag_helper(self, rhs, offset, naxes, extract, trace):
        ...

    @abstractmethod
    def eye(self, k) -> None:
        ...

    @abstractmethod
    def arange(self, start, stop, step) -> None:
        ...

    @abstractmethod
    def tile(self, rhs, reps) -> None:
        ...

    @abstractmethod
    def trilu(self, rhs, k, lower) -> None:
        ...

    @abstractmethod
    def bincount(self, rhs, weights=None) -> None:
        ...

    @abstractmethod
    def nonzero(self):
        ...

    @abstractmethod
    def bitgenerator_random_raw(
        self, bitgen, generatorType, seed, flags
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_integers(
        self, bitgen, generatorType, seed, flags, low, high
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_uniform(
        self, bitgen, generatorType, seed, flags, low, high
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_lognormal(
        self, bitgen, generatorType, seed, flags, mean, sigma
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_normal(
        self, bitgen, generatorType, seed, flags, mean, sigma
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_poisson(
        self, bitgen, generatorType, seed, flags, lam
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_exponential(
        self, bitgen, generatorType, seed, flags, scale
    ) -> None:
        ...

    @abstractmethod
    def bitgenerator_gumbel(
        self, bitgen, generatorType, seed, flags, mu, beta
    ) -> None:
        ...

    @abstractmethod
    def random_uniform(self) -> None:
        ...

    @abstractmethod
    def partition(
        self,
        rhs,
        kth,
        argpartition=False,
        axis=-1,
        kind="introselect",
        order=None,
    ) -> None:
        ...

    @abstractmethod
    def random_normal(self) -> None:
        ...

    @abstractmethod
    def random_integer(self, low, high) -> None:
        ...

    @abstractmethod
    def sort(
        self, rhs, argsort=False, axis=-1, kind="quicksort", order=None
    ) -> None:
        ...

    @abstractmethod
    def unary_op(self, op, rhs, where, args, multiout=None) -> None:
        ...

    @abstractmethod
    def unary_reduction(
        self,
        op,
        redop,
        rhs,
        where,
        orig_axis,
        axes,
        keepdims,
        args,
        initial,
    ):
        ...

    @abstractmethod
    def isclose(self, rhs1, rhs2, rtol, atol, equal_nan) -> None:
        ...

    @abstractmethod
    def binary_op(self, op, rhs1, rhs2, where, args) -> None:
        ...

    @abstractmethod
    def binary_reduction(self, op, rhs1, rhs2, broadcast, args):
        ...

    @abstractmethod
    def where(self, op, rhs1, rhs2, rhs3):
        ...

    @abstractmethod
    def cholesky(self, src, no_tril) -> None:
        ...

    @abstractmethod
    def unique(self):
        ...

    @abstractmethod
    def create_window(self, op_code, *args) -> None:
        ...
