from typing import Any

class CFunc(object):
    def __init__(
        self, pyfunc: Any, sig: Any, locals: Any, options: Any
    ) -> None: ...
    @property
    def address(self) -> int: ...
