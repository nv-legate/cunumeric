import os, sys, time, functools
import numpy as np
from quantile_impl import *

def f_test_axis_none(method, str_method, arr, qs_arr, eps = 1.0e-8):
    # cunumeric:
    #print("cunumeric:")
    q_out = quantile(arr, qs_arr, axis = None, method = str_method)
    #print(q_out)

    # np:
    #print("numpy:")
    np_q_out = np.quantile(arr, qs_arr, axis = None, method = str_method)
    #print(np_q_out)

    qo_flat = q_out.flatten()
    np_qo_flat = np_q_out.flatten()
    sz = qo_flat.size
    eq_hyp = all([np.abs(qo_flat[i] - np_qo_flat[i]) < eps for i in range(0, sz)])

    # print("Passed!" if eq_hyp else "Failed")
    # assert eq_hyp
    
    return (eq_hyp, q_out, np_q_out)


def f_test_axis(method, str_method, arr, axis, qs_arr, eps = 1.0e-8, keepdims = False):
    # cunumeric:
    #print("cunumeric axis = %d:"%(axis))
    q_out = quantile(arr, qs_arr, axis = axis, method = str_method, keepdims = keepdims)
    #print(q_out)

    # np:
    #print("numpy axis = %d:"%(axis))
    np_q_out = np.quantile(arr, qs_arr, axis = axis, method = str_method, keepdims = keepdims)
    #print(np_q_out)

    qo_flat = q_out.flatten().astype(np.float64)
    np_qo_flat = np_q_out.flatten().astype(np.float64)
    sz = qo_flat.size
    eq_hyp = all([np.abs(qo_flat[i] - np_qo_flat[i]) < eps for i in range(0, sz)])
    
    # print("Passed!" if eq_hyp else "Failed")
    # assert eq_hyp

    return (eq_hyp, q_out, np_q_out)


if __name__ == "__main__":

    # numpy bug: averaged_inverted_cdf == inverted_cdf
    # (see also https://math.stackexchange.com/questions/4449830/quantile-type-2-averaged-inverted-cdf-returns-seemingly-inconsistent-results-i)
    #
    methods = [inverted_cdf, inverted_cdf, closest_observation, interpolated_inverted_cdf, hazen, weibull, linear, median_unbiased, normal_unbiased, lower, higher, midpoint, nearest]

    str_methods = ['inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 'median_unbiased', 'normal_unbiased', 'lower', 'higher', 'midpoint', 'nearest']

    # qs_arr = np.array([0.001], dtype = 'float64')
    # qs_arr = np.array([0.001, 0.37, 0.42, 0.67, 0.83, 0.99]) # passed...
    #
    # qs_arr = np.array([0.001, 0.37, 0.42, 0.67, 0.83, 0.99, 0.39, 0.49]) # passed... 
    
    # qs_arr = np.ndarray(shape = (2,3), buffer = np.array([0.001, 0.37, 0.42, 0.67, 0.83, 0.99]).data) # passed...

    # qs_arr = np.ndarray(shape = (2,4), buffer = np.array([0.001, 0.37, 0.42, 0.5, 0.67, 0.83, 0.99, 0.39]).data) # fails, 0.5 seems to be the issue!

    # qs_arr = np.ndarray(shape = (2,4), buffer = np.array([0.001, 0.37, 0.42, 0.67, 0.83, 0.99, 0.39, 0.49]).data) # passed...

    qs_arr = np.array([0.001, 0.37, 0.42, 0.67, 0.83, 0.99, 0.39, 0.49 , 0.5]) # passes... for method != 2
    #
    # qs_arr = np.array([0.001, 0.37, 0.42, 0.67, 0.83, 0.99, 0.39, 0.49]) # passed...

    # scalar test:
    # qs_arr = 0.5 # passed...
    
    arr1 = np.array([1.0, 0.13, 2.11, 1.9, 9.2]) # passed...
    # sorted == [0.13, 1.0, 1.9, 2.11, 9.2]
    
    arr2 = np.ndarray(shape = (2,3), buffer = np.array([1.0, 0.13, 2.11, 1.9, 9.2, 0.17]).data) # passed...

    arr234 = np.ndarray(shape = (2,3,4), buffer = np.array([1,2,2,40,1,1,2,1,0,10,3,3,40,15,3,7,5,4,7,3,5,1,0,9]), dtype = int)

    # arr = arr1 # passed...
    # qs_arr = np.array([0.001, 0.5]) # passed...
    
    # arr = arr2 # passed...

    # keepdims = True # passed...
    keepdims = False # passed...

    # arr = arr2 # fails with q = 0.5 ...
    # axes = 0 # passed...
    
    # axes = (0,2) # passed...
    arr = arr234 # passed... if q != 0.5 or method != 2

    # fails for 2nd method:
    #
    # arr3 = np.array([0.13, 2.11])
    # arr = arr3
    # qs_arr = 0.5

    axes = 0
    
    for i in range(0, len(methods)):
        res = f_test_axis(methods[i], str_methods[i], arr, axes, qs_arr, keepdims = keepdims)
        str_res = "Passed!" if res[0] else "Failed"
        print("%s, %s, %s"%(str_methods[i], "axis=0", str_res))
        if res[0] == False:
            print("cu.%s"%(str(res[1])))
            print("np.%s"%(str(res[2])))
        
        res = f_test_axis_none(methods[i], str_methods[i], arr, qs_arr)
        str_res = "Passed!" if res[0] else "Failed"
        print("%s, %s, %s"%(str_methods[i], "axis=None", str_res))
        if res[0] == False:
            print("cu.%s"%(str(res[1])))
            print("np.%s"%(str(res[2])))
            
        if len(arr.shape) > 1:
            res = f_test_axis(methods[i], str_methods[i], arr, 1, qs_arr, keepdims = keepdims)
            str_res = "Passed!" if res[0] else "Failed"
            print("%s, %s, %s"%(str_methods[i], "axis=1", str_res))
            if res[0] == False:
                print("cu.%s"%(str(res[1])))
                print("np.%s"%(str(res[2])))
            
    print("Done!\n")
