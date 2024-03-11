from typing import Any, Optional, Tuple

from numba.core.codegen import Codegen, CodeLibrary

class CUDACodeLibrary(CodeLibrary):
    codegen: "JITCUDACodegen"
    name: str

    def get_asm_str(self, cc: Optional[Tuple[int, int]] = None) -> str: ...

class JITCUDACodegen(Codegen):
    def create_library(self, name: str, **kwargs: Any) -> CUDACodeLibrary: ...
