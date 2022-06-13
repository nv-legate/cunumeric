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

from .array import add_boilerplate
from .module import empty


@add_boilerplate("a")
def packbits(a, axis=None, bitorder="big"):
    """

    Packs the elements of a binary-valued array into bits in a uint8 array.

    The result is padded to full bytes by inserting zero bits at the end.

    Parameters
    ----------
    a : array_like
        An array of integers or booleans whose elements should be packed to
        bits.
    axis : int, optional
        The dimension over which bit-packing is done.
        ``None`` implies packing the flattened array.
    bitorder : ``{"big", "little"}``, optional
        The order of the input bits. 'big' will mimic bin(val),
        ``[0, 0, 0, 0, 0, 0, 1, 1] => 3 = 0b00000011``, 'little' will
        reverse the order so ``[1, 1, 0, 0, 0, 0, 0, 0] => 3``.
        Defaults to "big".

    Returns
    -------
    packed : ndarray
        Array of type uint8 whose elements represent bits corresponding to the
        logical (0 or nonzero) value of the input elements. The shape of
        `packed` has the same number of dimensions as the input (unless `axis`
        is None, in which case the output is 1-D).

    See Also
    --------
    numpy.packbits

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if axis is None:
        if a.ndim > 1:
            a = a.ravel()
        sanitized_axis = 0
    elif axis < 0:
        sanitized_axis = axis + a.ndim
    else:
        sanitized_axis = axis

    if sanitized_axis < 0 or sanitized_axis >= a.ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {a.ndim}"
        )

    if bitorder not in ("big", "little"):
        raise ValueError("'order' must be either 'little' or 'big'")

    if a.dtype.kind not in ("u", "i", "b"):
        raise TypeError(
            "Expected an input array of integer or boolean data type"
        )

    out_shape = tuple(
        (extent + 7) // 8 if dim == sanitized_axis else extent
        for dim, extent in enumerate(a.shape)
    )
    out = empty(out_shape, dtype="B")
    out._thunk.packbits(a._thunk, sanitized_axis, bitorder)

    return out
