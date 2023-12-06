from typing import Any, Callable, Dict, Optional, Tuple, Union

from numba.core.compiler import CompileResult
from numba.core.types import Type

def compile_ptx(
    pyfunc: Callable[[Any], Any],
    args: Any,
    debug: bool = False,
    lineinfo: bool = False,
    device: bool = False,
    fastmath: bool = False,
    cc: Optional[Any] = None,
    opt: bool = True,
) -> tuple[Any]: ...
def compile_cuda(
    pyfunc: Callable[[Any], Any],
    return_type: Type,
    args: Tuple[Type, ...],
    debug: bool = False,
    lineinfo: bool = False,
    inline: bool = False,
    fastmath: bool = False,
    nvvm_options: Optional[Dict[str, Optional[Union[str, int]]]] = None,
    cc: Optional[Tuple[int, int]] = None,
) -> CompileResult: ...
