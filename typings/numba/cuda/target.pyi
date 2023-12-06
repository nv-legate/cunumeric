from llvmlite import ir
from numba.core.base import BaseContext
from numba.core.callconv import BaseCallConv

class CUDACallConv(BaseCallConv): ...

class CUDATargetContext(BaseContext):
    call_conv: CUDACallConv

    def create_module(self, name: str) -> ir.Module: ...
