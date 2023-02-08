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

import legate.core.types as ty
import numba
import numba.core.ccallback 
import numpy as np
import six

from cunumeric.runtime import runtime

from .array import convert_to_cunumeric_ndarray
from .config import CuNumericOpCode

# import numba.cuda
# import numba.types


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
        self._numba_func: Callable[[Any], Any]
        self._cpu_func: numba.core.ccallback.CFunc
        self._gpu_func: tuple[Any]
        self._otypes = None
        self._result = None
        self._args: List[Any] = []
        self._kwargs: List[Any] = []
        self._context = runtime.legate_context

        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc

        #FIXME
        if otypes is not None:
            raise NotImplementedError("Otypes variables are not supported yet")

        #FIXME
        if excluded is not None:
            raise NotImplementedError(
                "excluded variables are not supported yet"
            )
        #FIXME
        if cache:
            raise NotImplementedError("cache variable is not supported yet")

        #FIXME
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

    def _build_gpu_function(self) -> Any:

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
            ty = getattr(numba.core.types, ty)
            ty = numba.core.types.CPointer(ty)
            types.append(ty)
        return types

    def _compile_func_gpu(self) -> tuple[Any]:
        types = self._get_numba_types()
        arg_types = types + [numba.core.types.uint64]
        sig = (*arg_types,)

        cuda_arch = numba.cuda.get_current_device().compute_capability
        return numba.cuda.compile_ptx(self._numba_func, sig, cc=cuda_arch)

    def _compile_func_cpu(self) -> numba.core.ccallback.CFunc:
        sig = numba.core.types.void(
            numba.types.CPointer(numba.types.voidptr), numba.core.types.uint64
        )  # type: ignore

        return numba.cfunc(sig)(self._numba_func)

    def _execute_gpu(self) -> None:
        task = self._context.create_auto_task(CuNumericOpCode.EVAL_UDF)
        task.add_scalar_arg(self._gpu_func[0], ty.string)
        idx = 0
        a0 = self._args[0]._thunk
        a0 = runtime.to_deferred_array(a0)
        for a in self._args:
            a_tmp = runtime.to_deferred_array(a._thunk)
            task.add_input(a_tmp.base)
            task.add_output(a_tmp.base)
            if idx != 0:
                task.add_alignment(a0.base, a_tmp.base)
            idx += 1
            task.add_broadcast(
                a_tmp.base, axes=tuple(range(1, len(a_tmp.base.shape)))
            )
        task.execute()

    def _execute_cpu(self) -> None:
        task = self._context.create_auto_task(CuNumericOpCode.EVAL_UDF)
        task.add_scalar_arg(self._cpu_func.address, ty.uint64)  # type : ignore
        idx = 0
        a0 = self._args[0]._thunk
        a0 = runtime.to_deferred_array(a0)
        for a in self._args:
            a_tmp = runtime.to_deferred_array(a._thunk)
            task.add_input(a_tmp.base)
            task.add_output(a_tmp.base)
            if idx != 0:
                task.add_alignment(a0.base, a_tmp.base)
            idx += 1
            task.add_broadcast(
                a_tmp.base, axes=tuple(range(1, len(a_tmp.base.shape)))
            )
        task.execute()

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

        #        #FIXME: comment out when brodcast PR is merged
        #        #bring all argumants to the same shape and type:
        #        if len(self._args)>0:
        #             ty = self._args[0].dtype
        #             #FIXME: should we bring them all to the same type?
        #             for a in self._args:
        #                 if a.dtype != ty:
        #                    return TypeError("all arguments of "
        #                         "user defined function "
        #                      "should have the same type")

        #    shapes = tuple(a.shape for a in self._args)
        #    shape = broadcast_shapes(shapes)
        #    new_args = tuple()
        #    for a in self._args:
        #        a_new = a.broadcast_to(shape)
        #        new_args +=(a_new,)
        #    self._args = new_args

        self._kwargs = list(kwargs)
        if len(self._kwargs) > 1:
            raise NotImplementedError(
                "kwargs are not supported in user functions"
            )

        if runtime.num_gpus > 0:
            self._numba_func = self._build_gpu_function()
            self._gpu_func = self._compile_func_gpu()
            self._execute_gpu()
        else:
            self._numba_func = self._build_cpu_function()
            self._cpu_func = self._compile_func_cpu()
            self._execute_cpu()
