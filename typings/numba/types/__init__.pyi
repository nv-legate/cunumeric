                      configuration locations on your computer.

class Type(): ...

class Number(): ...

class Integer(Number):
    def __init__(self, name: str) ->None: ...

class CPointer (Type):
    def __init__ (self, dtype : Type) -> None : ...

uint32 = Integer('uint32')
uint64 = Integer('uint64')
void = none