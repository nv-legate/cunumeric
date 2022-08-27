import argparse

from legate.timing import time
import cunumeric as cn
import numpy as np

def tuple_dims(strings):
    return tuple(map(int, strings.split(",")))

def run_benchmark(fxn, arr, iters: int):
    cn_arr = cn.array(arr)
    out = fxn(cn_arr)
    start = time()
    for i in range(1, iters):
        fxn(cn_arr, out=out)
    stop = time()
    ms_per = (stop - start) / (iters-1) / 1e3
    bytes_accessed = arr.size * arr.itemsize
    print(bytes_accessed)
    tput_gb = bytes_accessed / ms_per / 1e6
    print("%20s: %12.8f ms/iteration" % (fxn.__name__, ms_per))
    print("%20s: %12.8f GB/s" % (fxn.__name__, tput_gb))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=10,
        dest="iters",
        help="number of iterations to run"
    )
    
    parser.add_argument(
        "-n",
        "--shape",
        type=tuple_dims,
        default=(100_000_000,),
        dest="shape",
        help="size of input array"
    )

    args = parser.parse_args()
    arr = np.random.randn(*args.shape).astype(np.float32)
    print("N=%dM" % (arr.size / 1e6))
    run_benchmark(cn.sum, arr, args.iters)

    run_benchmark(cn.argmax, arr, args.iters)
