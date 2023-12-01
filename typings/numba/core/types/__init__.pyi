class Opaque: ...

class NoneType(Opaque):
    def __init__(self, name: str) -> None: ...

class Type:
    def __init__(self, name: str) -> None: ...

class Number(Type): ...

class Integer(Number):
    def __init__(self, name: str) -> None: ...

class RawPointer:
    def __init__(self, name: str) -> None: ...

class CPointer(Type):
    def __init__(self, dtype: Type) -> None: ...

none = NoneType("none")

uint32 = Integer("uint32")
uint64 = Integer("uint64")
void = none
voidptr = Type("void*")
