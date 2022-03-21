import numpy as np

import cunumeric

from cunumeric.config import CuNumericOpCode
from cunumeric import runtime

# class BitGenerator:
#     DEFAULT = 0
#     XORWOW = 1
#     MRG32K3A = 2
#     MTGP32 = 3
#     MT19937 = 4
#     PHILOX4_32_10 = 5
    
#     __slots__ = [
#         "handle", # handle to the runtime id
#     ]

#     def __init__(self, seed=None, generatorType=DEFAULT):
#         task = runtime.legate_context.create_task(
#             CuNumericOpCode.CUNUMERIC_BITGENERATOR,
#             manual=True,
#             launch_domain=Rect(lo=(0,), hi=(self.num_gpus,)),

#         )
#         self.handle = runtime.get_generator_id()
#         task.add_input(self.handle)
#         task.add_input(generatorID)
        
#         task.execute()
#         runtime.legate_runtime.issue_execution_fence(block=True)

#     def random_raw(self, size=None, output=True):
#         raise NotImplementedError('Not Implemented')

def test_bitgenerator():
    # print(dir(cunumeric))
    a = cunumeric.random.BitGenerator()
    print("DONE")
    pass

if __name__ == "__main__":
    test_bitgenerator()