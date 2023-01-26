# Copyright 2023  NVIDIA Corporation
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

import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Union

# numba doesn't seem to include type hints
import numba.cuda  # type: ignore
import numba.types  # type: ignore
import numpy as np
import six

from cunumeric.runtime import runtime

from .array import convert_to_cunumeric_ndarray

_EXTERNAL_REFERENCE_PREFIX = "__extern_ref__"
_MASK_VAR = "__mask__"
_SIZE_VAR = "__size__"
_LOOP_VAR = "__i__"
_ARGS_VAR = "__args__"


class vectorize:
    """
    vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False,
              signature=None)
    Generalized function class.
    Define a vectorized function which takes a nested sequence of objects or
    numpy arrays as inputs and returns a single numpy array or a tuple of numpy
    arrays. The vectorized function evaluates `pyfunc` over successive tuples
    of the input arrays like the python map function, except it uses the
    broadcasting rules of numpy.
    The data type of the output of `vectorized` is determined by calling
    the function with the first element of the input.  This can be avoided
    by specifying the `otypes` argument.

    Parameters
    ----------
    pyfunc : callable
        A python function or method.
    otypes : str or list of dtypes, optional
        The output data type. It must be specified as either a string of
        typecode characters or a list of data type specifiers. There should
        be one data type specifier for each output.
    doc : str, optional
        The docstring for the function. If None, the docstring will be the
        ``pyfunc.__doc__``.
    excluded : set, optional
        Set of strings or integers representing the positional or keyword
        arguments for which the function will not be vectorized.  These will be
        passed directly to `pyfunc` unmodified.
    cache : bool, optional
        If `True`, then cache the first function call that determines
        the number of outputs if `otypes` is not provided.
    signature : string, optional
        Generalized universal function signature, e.g., ``(m,n),(n)->(m)`` for
        vectorized matrix-vector multiplication. If provided, ``pyfunc`` will
        be called with (and expected to return) arrays with shapes given by the
        size of corresponding core dimensions. By default, ``pyfunc`` is
        assumed to take scalars as input and output.

    Returns
    -------
    vectorized : callable
        Vectorized function.

    See Also
    --------
    numpy.vectorize

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    def __init__(
        self,
        pyfunc: Callable[[Any], Any],
        otypes: Optional[Union[str, list[Any]]] = None,
        doc: Optional[str] = None,
        excluded: Optional[set[Any]] = None,
        cache: Optional[bool] = False,
        signature: Optional[str] = None,
    ) -> None:
        self._pyfunc = pyfunc
        self._numba_func: Optional[Callable[[Any], Any]] = None
        self._device_func: Optional[Callable[[Any], Any]] = None
        self._otypes = None
        self._result = None
        self._args: List[Any] = []
        self._kwargs: List[Any] = []

        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc

        if otypes is not None:
            raise NotImplementedError("Otypes variables are not supported yet")

        if excluded is not None:
            raise NotImplementedError(
                "excluded variables are not supported yet"
            )
        if cache:
            raise NotImplementedError("cache variable is not supported yet")

        if signature is not None:
            raise NotImplementedError(
                "signature variable is not supported yet"
            )

        # FIXME check return of the user function
        # return annotation (we supprt only void)

    #        if inspect.signature(self._pyfunc).return_annotation()
    #            != inspect._empty:
    #            raise NotImplementedError(
    #                "user defined functions can't have a return"
    #            )

    def _get_func_body(self, func: Callable[[Any], Any]) -> list[str]:
        """Using the magic method __doc__, we KNOW the size of the docstring.
        We then, just substract this from the total length of the function
        """
        lines_to_skip = 0
        if func.__doc__ is not None and len(func.__doc__.split("\n")) > 0:
            lines_to_skip = len(func.__doc__.split("\n"))

        lines = inspect.getsourcelines(func)[0]

        return_lines = []
        for i in range(lines_to_skip + 1, len(lines)):
            return_lines.append(lines[i].rstrip())
        return return_lines

    def _build_gpu_function(self) -> Callable[[Any], Any]:

        funcid = "vectorized_{}".format(self._pyfunc.__name__)

        # Preamble
        lines = ["from numba import cuda"]

        # Signature
        argnames = list(k for k in inspect.signature(self._pyfunc).parameters)
        args = argnames + [_SIZE_VAR]
        lines.append("def {}({}):".format(funcid, ",".join(args)))

        # Initialize the index variable and return immediately
        # when it exceeds the data size
        lines.append("    {} = cuda.grid(1)".format(_LOOP_VAR))
        lines.append("    if {} >= {}:".format(_LOOP_VAR, _SIZE_VAR))
        lines.append("        return")

        # Kernel body
        def _lift_to_array_access(m: Any) -> str:
            name = m.group(0)
            if name in argnames:
                return "{}[{}]".format(name, _LOOP_VAR)
            else:
                return "{}".format(name)

        # kernel body
        lines_old = self._get_func_body(self._pyfunc)
        for line in lines_old:
            l_new = re.sub(r"[_a-z]\w*", _lift_to_array_access, line)
            lines.append(l_new)

        # Evaluate the string to get the Python function
        body = "\n".join(lines)
        glbs: Dict[str, Any] = {}
        six.exec_(body, glbs)
        return glbs[funcid]

    def _build_cpu_function(self) -> Callable[[Any], Any]:

        funcid = "vectorized_{}".format(self._pyfunc.__name__)

        # Preamble
        lines = ["from numba import carray, types"]

        # Signature
        lines.append("def {}({}, {}):".format(funcid, _ARGS_VAR, _SIZE_VAR))

        # Unpack kernel arguments
        def _emit_assignment(
            var: Any, idx: int, sz: Any, ty: np.dtype[Any]
        ) -> None:
            lines.append(
                "    {} = carray({}[{}], {}, types.{})".format(
                    var, _ARGS_VAR, idx, sz, ty
                )
            )

        # get names of arguments
        argnames = list(k for k in inspect.signature(self._pyfunc).parameters)
        arg_idx = 0
        for a in self._args:
            ty = a.dtype
            _emit_assignment(argnames[arg_idx], arg_idx, _SIZE_VAR, ty)
            arg_idx += 1

        # Main loop
        lines.append("    for {} in range({}):".format(_LOOP_VAR, _SIZE_VAR))

        lines_old = self._get_func_body(self._pyfunc)

        def _lift_to_array_access(m: Any) -> str:
            name = m.group(0)
            if name in argnames:
                return "{}[{}]".format(name, _LOOP_VAR)
            else:
                return "{}[0]".format(name)

        # lines_new = []
        for line in lines_old:
            l_new = re.sub(r"[_a-z]\w*", _lift_to_array_access, line)
            lines.append("        " + l_new)

        # Evaluate the string to get the Python function
        body = "\n".join(lines)
        glbs: Dict[str, Any] = {}
        six.exec_(body, glbs)
        return glbs[funcid]

    def _get_numba_types(self, need_pointer: bool = True) -> list[Any]:
        types = []
        for arg in self._args:
            ty = arg.dtype
            ty = str(ty) if ty != bool else "int8"
            ty = getattr(numba.types, ty)
            ty = numba.types.CPointer(ty)
            types.append(ty)
        return types

    def _compile_func_gpu(self) -> Callable[[Any], Any]:
        types = self._get_numba_types()
        arg_types = types + [numba.types.uint64]
        sig = (*arg_types,)

        cuda_arch = numba.cuda.get_current_device().compute_capability
        return numba.cuda.compile_ptx(self._numba_func, sig, cc=cuda_arch)

    def _compile_func_cpu(self) -> Any:
        sig = numba.types.void(
            numba.types.CPointer(numba.types.voidptr), numba.types.uint64
        )

        return numba.cfunc(sig)(self._numba_func)

    #     def _execute_gpu(self):
    #        task = self.context.create_auto_task(CuNumericOpCode.LOAD_PTX)
    #        task..add_future(
    #            self._runtime.create_future_from_string(self._device_func)
    #        )
    #        kernel_fun = task.execute()

    #        task = self.context.create_auto_task(CuNumericOpCode.EVAL_UDF)
    # This will be ignored
    #        task.add_scalar_arg(0, ty.uint64)
    #        task.add_future_map(kernel_fun)
    #        task.execute()

    #     def _execute_cpu(self):

    #        task = self.context.create_auto_task(CuNumericOpCode.EVAL_UDF)
    #        task.add_scalar_arg(self._device_func.address, ty.uint64)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """
        Return arrays with the results of `pyfunc` broadcast (vectorized) over
        `args` and `kwargs` not in `excluded`.
        """
        self._args = list(
            convert_to_cunumeric_ndarray(arg) if arg is not None else arg
            for (idx, arg) in enumerate(args)
        )
        for arg in self._args:
            if arg is None:
                raise ValueError(
                    "None is not supported in user function "
                    "passed to cunumeric.vectorize"
                )

        self._kwargs = list(kwargs)
        if len(self._kwargs) > 1:
            raise NotImplementedError(
                "kwargs are not supported in user functions"
            )

        if runtime.num_gpus > 0:
            self._numba_func = self._build_gpu_function()
            self._device_func = self._compile_func_gpu()
        #            self._execute_gpu()
        else:
            self._numba_func = self._build_cpu_function()
            self._device_func = self._compile_func_cpu()
        #            self._execute_cpu()

        return self._result
