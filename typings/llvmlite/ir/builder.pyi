from typing import Iterable, Optional, Union

from llvmlite.ir.instructions import Instruction, Ret
from llvmlite.ir.values import Block, Value

class IRBuilder:
    def __init__(self, block: Optional[Block]): ...
    def ret(self, return_value: Value) -> Ret: ...
    def extract_value(
        self,
        agg: Value,
        idx: Union[Iterable[int], int],
        name: Optional[str] = "",
    ) -> Instruction: ...
    def store(
        self, value: Value, ptr: Value, align: Optional[int] = None
    ) -> Instruction: ...
    def ret_void(self) -> Ret: ...
