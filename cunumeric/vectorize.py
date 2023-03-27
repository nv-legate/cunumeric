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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import legate.core.types as ty
import numba
import numba.core.ccallback
import numpy as np
from legate.core import Rect, track_provenance

from cunumeric.runtime import runtime

from .array import convert_to_cunumeric_ndarray
from .config import CuNumericOpCode
from .module import full
from .utils import convert_to_cunumeric_dtype

_EXTERNAL_REFERENCE_PREFIX = "__extern_ref__"
_MASK_VAR = "__mask__"
_SIZE_VAR = "__size__"
_LOOP_VAR = "__i__"
_ARGS_VAR = "__args__"
_DIM_VAR = "__dim__"
_STRIDES_VAR = "__strides__"
_PITCHES_VAR = "__pitches__"


class vectorize:
    def __init__(
        self,
        pyfunc: Callable[[Any], Any],
        otypes: Optional[Union[str, list[Any]]] = None,
        doc: Optional[str] = None,
        excluded: Optional[set[Any]] = None,
        cache: bool = False,
        signature: Optional[str] = None,
    ) -> None:
        """
        vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False,
                  signature=None)

        Generalized function class.
        Define a vectorized function which takes a nested sequence of
        objects or numpy arrays as inputs and returns a single numpy array
        or a tuple of numpy arrays.
        User defined pyfunction will be executed in a single cuNumeric task
        over a set of arguments.
        The data type of the output of `vectorized` is determined by calling
        the function with the first element of the input.  This can be avoided
        by specifying the `otypes` argument.
        WARNING: when running with OpenMP back-end, "vectorize" will fall-back
        to the serial CPU implementation

        Parameters
        ----------
        pyfunc : callable
            A python function or method.
        otypes : str or list of dtypes, optional
            The output data type. It must be specified as either a string of
            typecode characters or a list of data type specifiers. There should
            be one data type specifier for each output.
            WARNING: cuNumeric currently requires all output types to be the
            same
        doc : str, optional
            The docstring for the function. If None, the docstring will be the
            ``pyfunc.__doc__``.
        excluded : set, optional
            Set of strings or integers representing the positional or keyword
            arguments for which the function will not be vectorized.
            These will be passed directly to `pyfunc` unmodified.
            WARNING: cuNumeric doesn't suport this argument at the moment
        cache : bool, optional
            If `True`, then cache the first function call that generates C fun-
            ction or CUDA kernel. We recomment enabling caching in cuNumeric
            for better performance, when possible.
            WARNING: in the case when cache=True, cuNumeric will parse function
            signature and create C function or CUDA kernel only once. This
            means that types of arguments passed to the vectorized function
            (arrays, scalars etc) should be the same each time we call it.
        signature : string, optional
            Generalized universal function signature, e.g., ``(m,n),(n)->(m)``
            for vectorized matrix-vector multiplication. If provided,
            ``pyfunc`` will be called with (and expected to return)
            arrays with shapes given by the size of corresponding core
            dimensions. By default, ``pyfunc`` is assumed to take scalars
            as input and output.
            WARNING: cuNumeric doesn't suport this argument at the moment

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

        self._pyfunc = pyfunc
        self._otypes: Optional[tuple[Any]] = None
        self._cache: bool = cache
        self._numba_func: Callable[[Any], Any]
        self._cpu_func: numba.core.ccallback.CFunc
        self._gpu_func: tuple[Any]
        self._args: List[Any] = []
        self._scalar_args: List[Any] = []
        self._scalar_idxs: List[int] = []
        self._scalar_names: List[str] = []
        self._arg_names: List[str] = []
        self._context = runtime.legate_context
        self._created: bool = False
        self._func_body: List[str] = []

        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc

        self._return_names = self._get_return_arguments()
        self._num_outputs: int = len(self._return_names)
        self._return_args: List[Any] = []
        self._output_dtype: Optional[np.dtype[Any]] = None
        self._cached_dtype: Optional[np.dtype[Any]] = None
        self._cached_scalar_types: List[Any] = []

        if otypes is not None:
            if self._num_outputs != len(otypes):
                raise ValueError(
                    "number of types in otypes is not consistente"
                    " with the number of return values defined in pyfunc"
                )
            if len(otypes) > 1:
                for t in otypes:
                    if t != otypes[0]:
                        raise NotImplementedError(
                            "cuNumeric doesn't support variable types"
                            " in otypes"
                        )
                self._output_dtype = np.dtype(otypes[0])

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

    def _get_func_body(self, func: Callable[[Any], Any]) -> List[str]:
        """Using the magic method __doc__, we KNOW the size of the docstring.
        We then, just subtract this from the total length of the function
        """
        lines_to_skip = 0
        if func.__doc__ is not None and len(func.__doc__.split("\n")) > 0:
            lines_to_skip = len(func.__doc__.split("\n"))

        lines = inspect.getsourcelines(func)[0]  # type ignore

        return_lines = []
        for i in range(lines_to_skip + 1, len(lines)):
            return_lines.append(lines[i].rstrip())
        return return_lines

    def _get_return_arguments(self) -> List[str]:
        """
        Returns the list of names for return arrays/values
        """
        self._func_body = self._get_func_body(self._pyfunc)
        return_names = []
        for ln in self._func_body:
            if "return" in ln:
                ln = ln.replace("return", "")
                ln = ln.replace(" ", "")
                return_names = ln.split(",")
        return return_names

    def _replace_name(
        self, name: str, _LOOP_VAR: str, is_gpu: bool = False
    ) -> str:
        """
        add indices to the names of input/output arrays in the function body
        """
        if (name in self._arg_names) or (name in self._return_names):
            return "{}[int({})]".format(name, _LOOP_VAR)
        else:
            if is_gpu or ((not is_gpu) and not (name in self._scalar_names)):
                return "{}".format(name)
            else:
                return "{}[0]".format(name)

    def _build_gpu_function(self) -> Any:
        funcid = "vectorized_{}".format(self._pyfunc.__name__)

        # Preamble
        lines = ["from numba import cuda"]
        # we add math and numpy so user-defined functions can use them
        lines.append("import math")
        lines.append("import numpy")

        # Signature
        args = (
            self._return_names
            + self._arg_names
            + self._scalar_names
            + [_SIZE_VAR]
            + [_DIM_VAR]
            + [_PITCHES_VAR]
            + [_STRIDES_VAR]
        )

        lines.append("def {}({}):".format(funcid, ",".join(args)))

        # Initialize the index variable and return immediately
        # when it exceeds the data size
        lines.append("    local_i = cuda.grid(1)")
        lines.append("    if local_i >= {}:".format(_SIZE_VAR))
        lines.append("        return")
        # we compute index for sparse data access when using Legion's
        # pointer.
        # a[x][y][z]=a[x*strides[0] + y*strides[1] + z*strides[2]]
        lines.append("    {}:int = 0".format(_LOOP_VAR))
        lines.append("    for p in range({}-1):".format(_DIM_VAR))
        # FIXME make sure we compute index correct for all data types
        lines.append("        x=int(local_i/{}[p])".format(_PITCHES_VAR))
        lines.append(
            "        local_i = int(local_i%{}[p])".format(_PITCHES_VAR)
        )
        lines.append(
            "        {}+=int(x*{}[p])".format(_LOOP_VAR, _STRIDES_VAR)
        )
        lines.append(
            "    {}+=int(local_i*{}[{}-1])".format(
                _LOOP_VAR, _STRIDES_VAR, _DIM_VAR
            )
        )

        # this function is used to replace all array names with array[i]
        def _lift_to_array_access(m: Any) -> str:
            return self._replace_name(m.group(0), _LOOP_VAR, True)

        # kernel body
        lines_old = self._func_body
        for line in lines_old:
            if not ("return" in line):
                l_new = re.sub(r"[_a-zA-Z]\w*", _lift_to_array_access, line)
                lines.append(l_new)

        # Evaluate the string to get the Python function
        body = "\n".join(lines)
        glbs: Dict[str, Any] = {}
        exec(body, glbs)
        return glbs[funcid]

    def _build_cpu_function(self) -> Callable[[Any], Any]:
        funcid = "vectorized_{}".format(self._pyfunc.__name__)

        # Preamble
        lines = ["from numba import carray, types"]
        # we add math and numpy so user-defined functions can use them
        lines.append("import math")
        lines.append("import numpy")

        # Signature
        lines.append(
            "def {}({}, {}, {}, {}, {}):".format(
                funcid,
                _ARGS_VAR,
                _SIZE_VAR,
                _DIM_VAR,
                _PITCHES_VAR,
                _STRIDES_VAR,
            )
        )

        # Unpack kernel arguments
        def _emit_assignment(
            var: Any, idx: int, sz: Any, ty: np.dtype[Any]
        ) -> None:
            lines.append(
                "    {} = carray({}[{}], {}, types.{})".format(
                    var, _ARGS_VAR, idx, sz, ty
                )
            )

        # define pyfunc arguments as carrays
        arg_idx = 0
        for count, a in enumerate(self._return_args):
            type_a = a.dtype
            _emit_assignment(
                self._return_names[count], arg_idx, _SIZE_VAR, type_a
            )
            arg_idx += 1
        for count, a in enumerate(self._args):
            type_a = a.dtype
            _emit_assignment(
                self._arg_names[count], arg_idx, _SIZE_VAR, type_a
            )
            arg_idx += 1
        for count, a in enumerate(self._scalar_args):
            scalar_type = np.dtype(type(a).__name__)
            _emit_assignment(
                self._scalar_names[count], arg_idx, 1, scalar_type
            )
            arg_idx += 1

        # Main loop
        lines.append("    for local_i in range({}):".format(_SIZE_VAR))
        # we compute index for sparse data access when using Legion's
        # pointer.
        # a[x][y][z]=a[x*strides[0] + y*strides[1] + z*strides[2]]
        lines.append("        {}:int = 0".format(_LOOP_VAR))
        lines.append("        j:int = local_i")
        lines.append("        for p in range({}-1):".format(_DIM_VAR))
        lines.append("            x=int(j/{}[p])".format(_PITCHES_VAR))
        lines.append("            j = int(j%{}[p])".format(_PITCHES_VAR))

        lines.append(
            "            {}+=int(x*{}[p])".format(_LOOP_VAR, _STRIDES_VAR)
        )
        lines.append(
            "        {}+=int(j*{}[{}-1])".format(
                _LOOP_VAR, _STRIDES_VAR, _DIM_VAR
            )
        )

        lines_old = self._func_body

        # Kernel body
        def _lift_to_array_access(m: Any) -> str:
            return self._replace_name(m.group(0), _LOOP_VAR)

        for line in lines_old:
            if not ("return" in line):
                l_new = re.sub(r"[_a-zA-Z]\w*", _lift_to_array_access, line)
                lines.append("    " + l_new)

        # Evaluate the string to get the Python function
        body = "\n".join(lines)
        glbs: Dict[str, Any] = {}
        exec(body, glbs)
        return glbs[funcid]

    def _get_numba_types(self, need_pointer: bool = True) -> list[Any]:
        types = []
        for arg in self._return_args + self._args:
            type_a = arg.dtype
            type_a = str(type_a) if type_a != bool else "int8"
            type_a = getattr(numba.core.types, type_a)
            type_a = numba.core.types.CPointer(type_a)
            types.append(type_a)
        for arg in self._scalar_args:
            type_a = np.dtype(type(arg).__name__)
            type_a = str(type_a) if type_a != bool else "int8"
            type_a = getattr(numba.core.types, type_a)
            types.append(type_a)
        return types

    def _compile_func_gpu(self) -> tuple[Any]:
        types = self._get_numba_types()
        arg_types = (
            types
            + [numba.core.types.uint64]
            + [numba.core.types.uint64]
            + [numba.core.types.CPointer(numba.core.types.uint64)]
            + [numba.core.types.CPointer(numba.core.types.uint64)]
        )
        sig = (*arg_types,)

        cuda_arch = numba.cuda.get_current_device().compute_capability
        return numba.cuda.compile_ptx(self._numba_func, sig, cc=cuda_arch)

    def _compile_func_cpu(self) -> numba.core.ccallback.CFunc:
        sig = numba.core.types.void(  # type: ignore
            numba.types.CPointer(numba.types.voidptr),
            numba.core.types.uint64,
            numba.core.types.uint64,
            numba.core.types.CPointer(numba.core.types.uint64),
            numba.core.types.CPointer(numba.core.types.uint64),
        )

        return numba.cfunc(sig)(self._numba_func)

    def _create_cuda_kernel(self, num_gpus: int) -> None:
        # create CUDA kernel
        launch_domain = Rect(lo=(0,), hi=(num_gpus,))
        kernel_task = self._context.create_manual_task(
            CuNumericOpCode.CREATE_CU_KERNEL,
            launch_domain=launch_domain,
        )
        ptx_hash = hash(self._gpu_func[0])
        kernel_task.add_scalar_arg(ptx_hash, ty.int64)
        kernel_task.add_scalar_arg(self._gpu_func[0], ty.string)
        kernel_task.execute()
        # we want to make sure EVAL_UDF function is not executed before
        # CUDA kernel is created
        self._context.issue_execution_fence(block=True)

        # task has finished by the time we set self._created to True
        if self._cache:
            self._created = True

    @track_provenance(runtime.legate_context)
    def _execute(self, is_gpu: bool, num_gpus: int = 0) -> None:
        if is_gpu and not self._created:
            self._create_cuda_kernel(num_gpus)

        task = self._context.create_auto_task(CuNumericOpCode.EVAL_UDF)
        task.add_scalar_arg(self._num_outputs, ty.uint32)  # N of outputs
        task.add_scalar_arg(
            len(self._scalar_args), ty.uint32
        )  # N of scalar_args

        # add all scalar arguments first
        for a in self._scalar_args:
            dtype = convert_to_cunumeric_dtype(type(a).__name__)
            task.add_scalar_arg(a, dtype)

        num_args = len(self._args)
        # add return arguments with RW permissions
        first_array = None
        if self._num_outputs > 0:
            first_array = runtime.to_deferred_array(
                self._return_args[0]._thunk
            )
            task.add_input(first_array.base)
            task.add_output(first_array.base)

            for i in range(1, self._num_outputs):
                a_tmp = runtime.to_deferred_array(self._return_args[i]._thunk)
                a_tmp_base = a_tmp.base
                task.add_input(a_tmp_base)
                task.add_output(a_tmp_base)
                task.add_alignment(first_array.base, a_tmp_base)

        # add array arguments with read-only permissions
        if num_args > 0:
            start = 0
            if first_array is None:
                first_array = runtime.to_deferred_array(self._args[0]._thunk)
                task.add_input(first_array.base)
                start = 1
            for i in range(start, num_args):
                a_tmp = runtime.to_deferred_array(self._args[i]._thunk)
                a_tmp_base = a_tmp.base
                task.add_input(a_tmp_base)
                task.add_alignment(first_array.base, a_tmp_base)

        if is_gpu:
            ptx_hash = hash(self._gpu_func[0])
            task.add_scalar_arg(ptx_hash, ty.int64)
        else:
            task.add_scalar_arg(
                self._cpu_func.address, ty.uint64
            )  # type : ignore
        task.execute()

    def _filter_arguments_and_check(self) -> None:
        # this method will filter return and input arguments
        # it will also check shape and type of the arguments

        output_shape: Tuple[int] = (-1,)
        output_dtype = self._output_dtype
        self._return_args.clear()

        # if output type is not specified, we need to decide
        # which one to use
        # we also want to choose the shape for output array

        # check if output variable is in input arguments - >
        # then use it's dtype and shape
        for r in self._return_names:
            if r in self._arg_names:
                idx = self._arg_names.index(r)
                if output_dtype is None:
                    output_dtype = self._args[idx].dtype
                if output_shape == (-1,):
                    output_shape = self._args[idx].shape
                break

        # the case if we didn't find output argument in input argnames
        if output_shape == (-1,):
            for r in self._return_names:
                if r in self._scalar_names:
                    idx = self._scalar_names.index(r)
                    if output_dtype is None:
                        output_dtype = np.dtype(type(self._scalar_args[idx]))
                    output_shape = (1,)
                    break

        if self._cache and not (self._cached_dtype is None):
            if self._cached_dtype != output_dtype:
                raise TypeError(
                    "types of the arguments should stay the same"
                    " for each invocation of the vectorize object"
                )
        elif self._cache:
            self._cached_dtype = output_dtype

        # FIXME
        # we could find common type of input arguments here and
        # broadcasted shapes
        if self._num_outputs > 0 and output_dtype is None:
            raise ValueError("Unable to choose output dtype")
        if self._num_outputs > 0 and output_shape is None:
            raise ValueError("Unable to choose output shape")

        # filing the list of return arguments
        # check if there are return argnames in input argnames,
        # if not, create a new array
        for r in self._return_names:
            if r in self._arg_names:
                idx = self._arg_names.index(r)
                if self._args[idx].shape != output_shape:
                    raise ValueError(
                        "all output arrays should have the same shape"
                    )
                if output_dtype != self._args[idx].dtype:
                    runtime.warn(
                        "converting input array to output types in user func ",
                        category=RuntimeWarning,
                    )
                    self._args[idx] = self._args[idx].astype(output_dtype)
                self._return_args.append(self._args[idx])
                self._args.remove(self._args[idx])
                self._arg_names.remove(r)
            elif r in self._scalar_names:
                idx = self._scalar_names.index(r)
                if output_shape != (1,):
                    raise ValueError(
                        "all output arrays should have the same shape"
                    )
                self._return_args.append(
                    full(output_shape, self._scalar_args[idx], output_dtype)
                )
                self._scalar_args.remove(self._scalar_args[idx])
                self._scalar_names.remove(r)
            else:
                # create array and add it to the list of return_args
                tmp_ret = full(output_shape, 0, output_dtype)
                self._return_args.append(tmp_ret)

        # check types and shapes
        if len(self._args) > 0:
            for count, a in enumerate(self._args):
                if output_dtype != a.dtype:
                    runtime.warn(
                        "converting input array to output types in user func ",
                        category=RuntimeWarning,
                    )
                    self._args[count] = self._args[count].astype(output_dtype)
                # FIXME broadcast shapes
                if output_shape != self._args[count].shape:
                    raise ValueError(
                        "cuNumeric doesn't support "
                        "different shapes for arrays in "
                        "user function passed to vectorize"
                    )

    def __call__(self, *args: Any, **kwargs: Any) -> Union[Any, Tuple[Any]]:
        # each time we call `vectorize` on a pyfunc we need to clear
        # these lists to support different types of arguments passed
        self._scalar_args.clear()
        self._scalar_idxs.clear()
        self._args.clear()
        self._arg_names.clear()
        self._scalar_names.clear()

        scalar_idx = 0
        for i, arg in enumerate(args):
            if arg is None:
                raise ValueError(
                    "None is not supported in user function "
                    "passed to cunumeric.vectorize"
                )
            elif np.ndim(arg) == 0:
                if self._cache and not self._created:
                    self._cached_scalar_types.append(type(arg))
                elif self._cache:
                    if self._cached_scalar_types[scalar_idx] != type(arg):
                        raise TypeError(
                            "Input arguments to vectorized function should"
                            " have consistent types for each invocation"
                        )
                self._scalar_args.append(arg)
                self._scalar_idxs.append(i)
                scalar_idx += 1
            else:
                self._args.append(convert_to_cunumeric_ndarray(arg))

        # first fill arrays to argnames, then scalars:
        for i, k in enumerate(inspect.signature(self._pyfunc).parameters):
            if not (i in self._scalar_idxs):
                self._arg_names.append(k)

        for i, k in enumerate(inspect.signature(self._pyfunc).parameters):
            if i in self._scalar_idxs:
                self._scalar_names.append(k)

        if len(kwargs) > 0:
            raise NotImplementedError(
                "kwargs are not supported in user functions"
            )

        self._filter_arguments_and_check()

        if runtime.num_gpus > 0:
            if not self._created:
                self._numba_func = self._build_gpu_function()
                self._gpu_func = self._compile_func_gpu()
            self._execute(True, runtime.num_gpus)
        else:
            if not self._created:
                self._numba_func = self._build_cpu_function()
                self._cpu_func = self._compile_func_cpu()
                if self._cache:
                    self._created = True
            self._execute(False)

        if len(self._return_args) == 1:
            return self._return_args[0]
        if len(self._return_args) > 1:
            return tuple(self._return_args)
        return -1
