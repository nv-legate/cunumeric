from typing import Tuple, Union

from numba.core.types import Type
from numba.core.typing.signature import Signature

def normalize_signature(
    sig: Union[Tuple[Type, ...], str, Signature]
) -> Tuple[Tuple[Type, ...], Type]: ...
