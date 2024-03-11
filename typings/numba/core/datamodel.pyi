from typing import Sequence

from llvmlite.ir.types import Type

class ArgPacker:
    argument_types: Sequence[Type]
