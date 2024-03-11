from typing import Any, Callable

class _CompilerLock:
    def __call__(self, func: Callable[..., Any]) -> Callable[[Any], Any]: ...

global_compiler_lock = _CompilerLock()
