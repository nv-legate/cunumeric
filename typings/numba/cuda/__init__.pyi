from typing import Any

from numba.cuda.compiler import compile_ptx as compile_ptx

def get_current_device() -> Any: ...
