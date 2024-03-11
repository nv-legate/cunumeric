from typing import Any, Callable

import numba.cuda  # import compile_ptx
from numba.core import types
from numba.core.ccallback import CFunc
from numba.core.types import CPointer, uint64

def cfunc(sig: Any) -> Any:
    def wrapper(func: Callable[[Any], Any]) -> tuple[Any]: ...

__all__ = ["types"]
