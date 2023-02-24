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

import cProfile, pstats

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
from .utils import convert_to_cunumeric_dtype

# from legate.timing import time


# import numba.cuda
# import numba.types


_EXTERNAL_REFERENCE_PREFIX = "__extern_ref__"
_MASK_VAR = "__mask__"
_SIZE_VAR = "__size__"
_LOOP_VAR = "__i__"
_ARGS_VAR = "__args__"
_DIM_VAR = "__dim__"
_STRIDES_VAR = "__strides__"
_LO_POINT_VAR = "__lo_point__"
_PITCHES_VAR = "__pitches__"


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
        cache: bool = False,
        signature: Optional[str] = None,
    ) -> None:
        self._pyfunc = pyfunc
        self._numba_func: Callable[[Any], Any]
        self._cpu_func: numba.core.ccallback.CFunc
        self._gpu_func: tuple[Any]
        self._otypes: Optional[tuple[Any]] = None
        self._result = None
        self._args: List[Any] = []
        self._scalar_args: List[Any]=[]
        self._scalar_idxs:List[int]=[]
        self._scalar_names:List[str]=[]
        self._argnames:List[str]=[]
        self._kwargs: List[Any] = []
        self._context = runtime.legate_context
        self._created: bool = False
        self._cache: bool = cache
        self._num_outputs = 1  # there is at least 1 output

        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc

        if otypes is not None:
            self._num_outputs = len(otypes)
            if len(otypes) == 0:
                raise ValueError(
                    "There should be at least 1 type specified in otypes"
                )
            ty = otypes[0]
            for t in otypes:
                if t != ty:
                    raise NotImplementedError(
                        "cuNumeric doesn't support variable types in otypes"
                    )

        # FIXME
        if excluded is not None:
            raise NotImplementedError(
                "excluded variables are not supported yet"
            )

        # FIXME
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

    def _replace_name(self, name: str, _LOOP_VAR:str, is_gpu:bool=False) -> str:
        if name in self._argnames and not(name in self._scalar_names) :
            return "{}[{}]".format(name, _LOOP_VAR)
        elif name == "if":
            return "if "
        elif name == "return":
            return "return "
        elif name == "or":
            return "or "
        elif name == "and":
            return "and "
        elif name == "not":
            return "not "
        elif name == "min":
            return "min"
        elif name == "max":
            return "max"
        elif is_gpu:
            return "{}".format(name)
        else:
            return "{}[0]".format(name)


    def _build_gpu_function(self) -> Any:

        funcid = "vectorized_{}".format(self._pyfunc.__name__)

        # Preamble
        lines = ["from numba import cuda"]

        # Signature
        args = self._argnames + [_SIZE_VAR]+[_DIM_VAR]+[_PITCHES_VAR]+[_LO_POINT_VAR] +[_STRIDES_VAR]

        lines.append("def {}({}):".format(funcid, ",".join(args)))

        # Initialize the index variable and return immediately
        # when it exceeds the data size
        lines.append("    local_i = cuda.grid(1)")
        lines.append("    if local_i >= {}:".format(_SIZE_VAR))
        lines.append("        return")
        lines.append("    {}:int = 0".format(_LOOP_VAR))
        lines.append("    for p in range({}-1):".format(_DIM_VAR))
        lines.append("        x={}[p]+int(local_i/{}[p])".format(_LO_POINT_VAR,_PITCHES_VAR))
        lines.append("        local_i = local_i-{}[p]*int(local_i/{}[p])".format(_PITCHES_VAR,_PITCHES_VAR))
        lines.append("        {}+=int(x*{}[p])".format(_LOOP_VAR, _STRIDES_VAR))
        lines.append("    {}+=int(local_i*{}[{}-1])".format(_LOOP_VAR, _STRIDES_VAR, _DIM_VAR))

        # Kernel body
        def _lift_to_array_access(m: Any) -> str:
            return self._replace_name(m.group(0), _LOOP_VAR, True)

        # kernel body
        lines_old = self._get_func_body(self._pyfunc)
        for line in lines_old:
            l_new = re.sub(r"[_a-z]\w*", _lift_to_array_access, line)
            lines.append(l_new)
   
        #print("IRINA DEBUG GPU function",lines)

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
        arg_idx = 0
        for a in self._args:
            ty = a.dtype
            _emit_assignment(self._argnames[arg_idx], arg_idx, _SIZE_VAR, ty)
            arg_idx += 1
        for a in self._scalar_args:
            scalar_type = np.dtype(type(a).__name__)
            _emit_assignment(self._argnames[arg_idx], arg_idx, _SIZE_VAR, scalar_type)
            arg_idx += 1

        # Main loop
        lines.append("    for {} in range({}):".format(_LOOP_VAR, _SIZE_VAR))

        lines_old = self._get_func_body(self._pyfunc)

        # Kernel body
        def _lift_to_array_access(m: Any) -> str:
            return self._replace_name(m.group(0), _LOOP_VAR)

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
        for arg in self._scalar_args:
            ty = np.dtype(type(arg).__name__)
            ty = str(ty) if ty != bool else "int8"
            ty = getattr(numba.core.types, ty)
            types.append(ty)
        return types

    def _compile_func_gpu(self) -> tuple[Any]:
        types = self._get_numba_types()
        arg_types = types + [numba.core.types.uint64] + [numba.core.types.uint64]+[numba.core.types.CPointer(numba.core.types.uint64)]+ [numba.core.types.CPointer(numba.core.types.uint64)]+[numba.core.types.CPointer(numba.core.types.uint64)]
        sig = (*arg_types,)

        cuda_arch = numba.cuda.get_current_device().compute_capability
        return numba.cuda.compile_ptx(self._numba_func, sig, cc=cuda_arch)

    def _compile_func_cpu(self) -> numba.core.ccallback.CFunc:
        sig = numba.core.types.void(
            numba.types.CPointer(numba.types.voidptr), numba.core.types.uint64
        )  # type: ignore

        return numba.cfunc(sig)(self._numba_func)

    def _execute(self, is_gpu:bool) -> None:
        if is_gpu and not self._created:
            #create CUDA kernel
            kernel_task = self._context.create_auto_task(CuNumericOpCode.CREATE_CU_KERNEL)
            ptx_hash = hash(self._gpu_func[0])
            kernel_task.add_scalar_arg(ptx_hash, ty.int64)
            kernel_task.add_scalar_arg(self._gpu_func[0], ty.string)
            #adding unused array for creating correct launch domain
            #and set up dependency between kernel_task and task
            if len(self._args)>0:
               a0 = self._args[0]._thunk
               a0 = runtime.to_deferred_array(a0)
               kernel_task.add_input(a0.base)
               kernel_task.add_output(a0.base)
            kernel_task.execute()
                

        task = self._context.create_auto_task(CuNumericOpCode.EVAL_UDF)
        task.add_scalar_arg(self._num_outputs, ty.uint32)
        task.add_scalar_arg(len(self._scalar_args), ty.uint32)
         
        for a in self._scalar_args:
            dtype = convert_to_cunumeric_dtype(type(a).__name__)
            task.add_scalar_arg(a,dtype)

        if is_gpu:
            ptx_hash = hash(self._gpu_func[0])
            task.add_scalar_arg(ptx_hash, ty.int64)
            task.add_scalar_arg(self._created, bool)
        else:
            task.add_scalar_arg(self._cpu_func.address, ty.uint64)  # type : ignore
        a0 = self._args[0]._thunk
        a0 = runtime.to_deferred_array(a0)
        for count, a in enumerate(self._args):
            a_tmp = runtime.to_deferred_array(a._thunk)
            a_tmp=a_tmp.base
            task.add_input(a_tmp)
            if count < self._num_outputs:
                task.add_output(a_tmp)
            if count != 0:
                task.add_alignment(a0.base, a_tmp)
        task.execute()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """
        Return arrays with the results of `pyfunc` broadcast (vectorized) over
        `args` and `kwargs` not in `excluded`.
        """
        #profiler = cProfile.Profile()
        #profiler.enable()   
        if not self._created:
            self._scalar_args.clear()
            self._scalar_idxs.clear()
            self._args.clear()
            self._argnames.clear()
            self._scalar_names.clear()

            for i,arg in enumerate(args):
                if arg is None:
                    raise ValueError(
                        "None is not supported in user function "
                        "passed to cunumeric.vectorize"
                    )
                elif np.ndim(arg)==0:
                    self._scalar_args.append(arg)
                    self._scalar_idxs.append(i)
                else:
                    self._args.append(convert_to_cunumeric_ndarray(arg))

            #first fill arrays to argnames, then scalars:
            for i,k in enumerate(inspect.signature(self._pyfunc).parameters):
                if not(i in self._scalar_idxs):
                    self._argnames.append(k)

            for i,k in enumerate(inspect.signature(self._pyfunc).parameters):
                if i in self._scalar_idxs:
                    self._scalar_names.append(k)
                    self._argnames.append(k)

            self._kwargs = list(kwargs)
            if len(self._kwargs) > 1:
                raise NotImplementedError(
                    "kwargs are not supported in user functions"
                )

        #all output arrays should have the same type
        if len(self._args)>0:
            ty = self._args[0].dtype
            shape = self._args[0].shape
            for i in range (1, self._num_outputs):
                if ty!=self._args[i].dtype:
                    raise TypeError("cuNumeric doesnt support "
                        "different types for output data in "
                        "user function passed to vectorize")
                if shape != self._args[i].shape:
                    raise TypeError("cuNumeric doesnt support "
                        "different shapes for output data in "
                        "user function passed to vectorize")
            for i in range (self._num_outputs, len(self._args)):
                if ty!=self._args[i].dtype:
                    runtime.warn(
                        "converting input array to output types in user func ",
                        category=RuntimeWarning,
                    )
                    self._args[i] = self._args[i].astype(ty)
                if shape !=self._args[i].shape and np.ndim(self._args[i])>0:
                     raise TypeError("cuNumeric doesnt support "
                        "different shapes for arrays in "
                        "user function passed to vectorize")

        if runtime.num_gpus > 0:
            if not self._created:
                #print("IRINA DEBUG ptx is not created yet")
                self._numba_func = self._build_gpu_function()
                self._gpu_func = self._compile_func_gpu()
            #profiler = cProfile.Profile()
            #profiler.enable()
            self._execute(True)
            if not self._created and self._cache:
                self._created = True
            #profiler.disable()
            #stats = pstats.Stats(profiler).sort_stats('cumtime')
            #stats.print_stats()
        else:
            if not self._created:
                self._numba_func = self._build_cpu_function()
                self._cpu_func = self._compile_func_cpu()
                if self._cache:
                    self._created = True
            self._execute(False)

            
        #profiler.disable()
        #stats = pstats.Stats(profiler).sort_stats('cumtime')
        #stats.print_stats()


