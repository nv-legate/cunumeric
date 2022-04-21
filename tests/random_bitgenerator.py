# import numpy as np
# from cunumeric.array import ndarray

import cunumeric


def test_bitgenerator_type(t):
    print(f"testing for type = {t}")
    bitgen = t(seed=42)
    bitgen.random_raw(256, False)
    bitgen.random_raw((512, 256), False)
    r = bitgen.random_raw(256)  # deferred is None
    print(f"256 sum = {r.sum()}")
    r = bitgen.random_raw((1024, 1024))
    print(f"1k² sum = {r.sum()}")
    r = bitgen.random_raw(1024 * 1024)
    print(f"1M sum = {r.sum()}")
    bitgen = None
    print(f"DONE for type = {t}")


def test_bitgenerator_XORWOW():
    test_bitgenerator_type(cunumeric.random.XORWOW)


def test_bitgenerator_MRG32k3a():
    test_bitgenerator_type(cunumeric.random.MRG32k3a)


def test_bitgenerator_MTGP32():
    test_bitgenerator_type(cunumeric.random.MTGP32)


def test_bitgenerator_MT19937():
    test_bitgenerator_type(cunumeric.random.MT19937)


def test_bitgenerator_PHILOX4_32_10():
    test_bitgenerator_type(cunumeric.random.PHILOX4_32_10)


def test_eager_size_None():
    # https://github.com/nv-legate/cunumeric/pull/254#discussion_r846387540
    # TODO: how to test this ?
    # dar = ndarray((1), dtype=np.dtype(np.uint32))
    # ar = runtime.to_eager_array(dar)
    # ar.bitgenerator_random_raw(None)
    # print(str(ar[0]))
    pass


def test_size_None():
    # ERROR: Illegal request for pointer of non-dense rectangle
    # [ERROR] : accessor is not dense row major - DIM = 1
    #        -> out.accessor.is_dense_row_major returns false !
    rng = cunumeric.random.XORWOW()
    a = rng.random_raw()
    print(a)


if __name__ == "__main__":
    test_bitgenerator_MRG32k3a()
    test_bitgenerator_XORWOW()
    test_bitgenerator_PHILOX4_32_10()
    # test_bitgenerator_MTGP32()
    # test_bitgenerator_MT19937()
    # test_size_None()
