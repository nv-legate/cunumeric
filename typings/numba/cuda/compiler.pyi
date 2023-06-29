from typing import Any, Callable, Optional

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
