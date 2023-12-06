from typing import Iterable, Optional, Tuple

from llvmlite.ir.builder import IRBuilder
from llvmlite.ir.types import FunctionType, PointerType
from llvmlite.ir.values import Function, Value
from numba.core.base import BaseContext
from numba.core.datamodel import ArgPacker
from numba.core.types import Type

class BaseCallConv:
    def __init__(self, context: BaseContext): ...
    def _get_arg_packer(self, argtypes: Iterable[Type]) -> ArgPacker: ...
    def get_return_type(self, ty: Type) -> PointerType: ...
    def get_function_type(
        self, restype: Type, argtypes: Iterable[Type]
    ) -> FunctionType: ...
    def call_function(
        self,
        builder: IRBuilder,
        callee: Function,
        resty: Type,
        argtys: Iterable[Type],
        args: Iterable[Value],
        attrs: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[Value, Value]: ...
