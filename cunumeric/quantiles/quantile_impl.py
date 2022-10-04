# import numpy as np
import cunumeric as cu

# account for 0-based indexing
# there's no negative numbers
# arithmetic at this level,
# (pos, k) are always positive!
#
def floor_i(k):
    j = k-1 if k>0 else 0
    return int(j)

# Generic rule: if `q` input value falls onto a node, then return that node

# Discontinuous methods:
#
# 'inverted_cdf'
# q = quantile input \in [0, 1]
# n = sizeof(array)
#
# 'inverted_cdf'
# q = quantile input \in [0, 1]
# n = sizeof(array)
#
def inverted_cdf(q, n):
    pos = q*n
    k = cu.floor(pos)
    
    g = pos - k
    gamma = 1.0 if g > 0 else 0.0

    j = int(k) - 1
    if j < 0:
        return (0.0, 0)
    else:
        return (gamma, j)

# potential bug in numpy, here:
#
# 'averaged_inverted_cdf'
# potential bug in `numpy` here
#
def averaged_inverted_cdf(q, n):
    pos = q*n
    k = cu.floor(pos)
    
    g = pos - k
    gamma = 1.0 if g > 0 else 0.5

    j = int(k) - 1
    if j < 0:
        return (0.0, 0)
    else:
        return (gamma, j)

# 'closest_observation'
#
def closest_observation(q, n):
    ### p = q*n - 0.5
    ### pos = 0 if p < 0 else p

    # weird departure from paper
    # (bug?), but this fixes it:
    # also, j even in original paper
    # applied to 1-based indexing; we have 0-based!
    # numpy impl. doesn't account that the original paper used
    # 1-based indexing, 0-based j is still checked for evennes!
    # (see proof in quantile_policies.py)
    # 
    p0 = q*n - 0.5
    p = p0 - 1.0
    
    pos = 0 if p < 0 else p0
    k = cu.floor(pos)
        
    j = floor_i(k)
    gamma = 1 if k < pos else (0 if j%2 == 0 else 1)
    
    return (gamma, j)

# Continuous methods:
#
# Parzen method:
# 'interpolated_inverted_cdf'
#
def interpolated_inverted_cdf(q, n):
    pos = q*n
    k = cu.floor(pos)
    ### gamma = pos-k
    # this fixes it:
    #
    gamma = 0.0 if k == 0 else pos-k
    j = floor_i(k)
    return (gamma, j)

# Hazen method:
# 'hazen'
#
def hazen(q, n):
    pos = q*n + 0.5
    k = cu.floor(pos)
    # gamma = pos-k
    #
    # this fixes it:
    # (when pos > n: this actually selects the right point,
    #  which is the correct choice, because right = arr[n]
    #  gets invalidated)
    #
    gamma = 0.0 if (pos < 1 or pos > n) else pos-k
    
    j = floor_i(k)
    return (gamma, j)


# Weibull method:
# 'weibull'
#
def weibull(q, n):
    pos = q*(n+1)
    
    k = cu.floor(pos)
    # gamma = pos-k
    #
    # this fixes it:
    # (when pos > n: this actually selects the right point,
    #  which is the correct choice, because right = arr[n]
    #  gets invalidated)
    #
    gamma = 0.0 if (pos < 1 or pos > n) else pos-k
    
    j = floor_i(k)
    return (gamma, j)

# Gumbel method:
# 'linear'
#
def linear(q, n):
    pos = q*(n-1) + 1
    k = cu.floor(pos)
    #gamma = pos-k
    #
    # this fixes it:
    # (when pos > n: this actually selects the right point,
    #  which is the correct choice, because right = arr[n]
    #  gets invalidated)
    #
    gamma = 0.0 if (pos < 1 or pos > n) else pos-k
    
    j = floor_i(k)
    return (gamma, j)

# Johnson & Kotz method:
# 'median_unbiased'
#
def median_unbiased(q, n):
    fract = 1.0/3.0
    pos = q*(n+fract) + fract
    k = cu.floor(pos)
    
    # gamma = pos-k
    #
    # this fixes it:
    # (when pos > n: this actually selects the right point,
    #  which is the correct choice, because right = arr[n]
    #  gets invalidated)
    #
    gamma = 0.0 if (pos < 1 or pos > n) else pos-k
    
    j = floor_i(k)
    return (gamma, j)

# Blom method:
# 'normal_unbiased'
#
def normal_unbiased(q, n):
    fract1 = 0.25
    fract2 = 3.0/8.0
    pos = q*(n+fract1) + fract2
    k = cu.floor(pos)
    
    # gamma = pos-k
    #
    # this fixes it:
    # (when pos > n: this actually selects the right point,
    #  which is the correct choice, because right = arr[n]
    #  gets invalidated)
    #
    gamma = 0.0 if (pos < 1 or pos > n) else pos-k
    
    j = floor_i(k)
    return (gamma, j)


# `lower`
#
def lower(q, n):
    gamma = 0.0
    pos = q*(n-1)
    k = cu.floor(pos)

    j = int(k)
    return (gamma, j)


# `higher`
#
def higher(q, n):
    pos = q*(n-1)
    k = cu.floor(pos)

    # Generic rule: (k == pos)
    gamma = 0.0 if (pos == 0 or k == pos) else 1.0
    
    j = int(k)
    return (gamma, j)


# `midpoint`
#
def midpoint(q, n):
    pos = q*(n-1)
    k = cu.floor(pos)

    # Generic rule: (k == pos)
    gamma = 0.0 if (pos == 0 or k == pos) else 0.5
    
    j = int(k)
    return (gamma, j)


# `nearest`
#
def nearest(q, n):
    pos = q*(n-1)
    k = cu.floor(pos)

    gamma = 1.0 if pos > (cu.ceil(pos) + k)/2.0 else 0.0
    
    j = int(k)
    return (gamma, j)

# for the case when axis = tuple (non-singleton)
# reshuffling might have to be done (if tuple is non-consecutive)
# and the src array must be collapsed along that set of axes
#
# args:
#
# arr:    [in] source nd-array on which quantiles are calculated;
# axes_set: [in] tuple or list of axes (indices less than arr dimension);
#
# return: pair: (minimal_index, reshuffled_and_collapsed source array)
# TODO: check if reshuffling, reshaping is done in-place!
#
def reshuffle_reshape(arr, axes_set):
    is_sorted = lambda a: cu.all(a[:-1] <= a[1:])
    is_diff = lambda arr1, arr2 : len(arr1) != len(arr2) or cu.any([arr1[i] != arr2[i] for i in range(0, len(arr1))])

    ndim = len(arr.shape)
    
    if not is_sorted(axes_set):
        sorted_axes = tuple(cu.sort(axes_set))
    else:
        sorted_axes = tuple(axes_set)

    min_dim_index = sorted_axes[0]
    num_axes = len(sorted_axes)
    reshuffled_axes = tuple(range(min_dim_index, min_dim_index + num_axes))

    non_consecutive = is_diff(sorted_axes, reshuffled_axes)
    if non_consecutive:
        arr_shuffled = cu.moveaxis(arr, sorted_axes, reshuffled_axes)
    else:
        arr_shuffled = arr

    shape_reshuffled = arr_shuffled.shape
    collapsed_shape = cu.product([arr_shuffled.shape[i] for i in reshuffled_axes]) # WARNING: cuNumeric has not implemented numpy.product and is falling back to canonical numpy.

    redimed = tuple(range(0, min_dim_index+1)) + tuple(range(min_dim_index+num_axes, ndim))
    reshaped = tuple([collapsed_shape if k == min_dim_index else arr_shuffled.shape[k] for k in redimed])

    arr_reshaped = arr_shuffled.reshape(reshaped)
    return (min_dim_index, arr_reshaped)

# args:
#
# arr:      [in] source nd-array on which quantiles are calculated;
#                preccondition: assumed sorted!
# q_arr:    [in] quantile input values nd-array;
# axis:     [in] axis along which quantiles are calculated;
# method:   [in] func(q, n) returning (gamma, j),
#                where = array1D.size;
# keepdims: [in] boolean flag specifying whether collapsed axis should be kept as dim=1;
# to_dtype: [in] dtype to convert the result to;
# return: nd-array of quantile output values
#         where its shape is obtained as:
#         concatenating q_arr.shape with arr.shape \ {axis}
#         (the shape of `arr` obtained by taking the `axis` out)
#
def quantile_impl(arr, q_arr, axis, method, keepdims, to_dtype):

    ndims = len(arr.shape)
       
    if axis == None:
        n = arr.size
        remaining_shape = [] # only `q_arr` dictates shape;
                             # quantile applied to `arr` seen as 1D;
    else:
        n = arr.shape[axis]
        
        # arr.shape -{axis}; if keepdims use 1 for arr.shape[axis]: 
        # (can be empty [])
        #
        if keepdims:
            remaining_shape = [1 if k==axis else arr.shape[k] for k in range(0, ndims)]
        else:
            remaining_shape = [arr.shape[k] for k in range(0, ndims) if k != axis]
       
    # quantile interpolation method to act on vectors,
    # like an axpy call:
    # [gamma](vleft, vright) { return (1-gamma)*vleft + gamma*vright;}
    #
    linear_interp = lambda gamma, arr_lvals, arr_rvals: (1.0 - gamma)*arr_lvals + gamma*arr_rvals

    # helper:
    # create list of n repetitions of value `val`
    #
    repeat = lambda val, n : [val] * n

    # flattening to 1D might be the right way,
    # but may consider profiling other approaches
    # (slice assignment)
    #
    q_flat = q_arr.flatten().tolist()
       
    result_1D = [] # TODO: check if a container more efficient than list could be used to append results into
    #       e.g., pre-allocate 1d-array with len(q_flat) * prod(remaining_shape), where prod = foldl (\acc x -> acc*x) 1
    for q in q_flat:
        (gamma, j) = method(q,n) 
        (left_pos, right_pos) = (j, j+1) 

        # (N-1) dimensional ndarray of left, right
        # neighbor values:
        #
        arr_1D_lvals = arr.take([left_pos], axis).flatten()  # singleton list: [left_pos] ; extract values at index=left_pos;
        num_vals = len(arr_1D_lvals) # ( num_vals == prod(remaining_shape) ), provided prod = foldl (\acc x -> acc*x) 1, otherwise fails when remaining_shape != []

        if right_pos>=n:
            arr_1D_rvals = cu.array(repeat(0, num_vals)) # some quantile methods may result in j==(n-1), hence (j+1) could surpass array boundary;
        else:
            arr_1D_rvals = arr.take([right_pos], axis).flatten() # singleton list: [right_pos]; extract values at index=right_pos;

        ## gammmas     = repeat(gamma, num_vals) # need it only if `axpy` cannot take scalar `a`

        # this is the main source of parallelism
        # (the other one being on `q` inputs; ignored for now)
        # TODO: figure out what to pass to the `cunumeric` call
        #
        quantiles_ls = linear_interp(gamma, arr_1D_lvals, arr_1D_rvals) # fails with lists []! because `*` does something else; TODO: need cu.arrays here

        result_1D = [*result_1D, *quantiles_ls] # append

    # compose qarr.shape with arr.shape:
    #
    # result.shape = (q_arr.shape, arr.shape -{axis}):
    #
    qresult_shape = (*q_arr.shape, *remaining_shape)
    #
    # construct result NdArray:
    #


    # fails with cunumeric:
    # (numpy works)
    #
    # cunumeric BUG: TypeError: a bytes-like object is required, not 'ndarray'; TODO: open nvbug
    #
    # qs_all = cu.ndarray(shape = qresult_shape,
    #                     buffer = cu.array(result_1D),
    #                     dtype = to_dtype)
       
    qs_all = cu.asarray(result_1D, dtype = to_dtype).reshape(qresult_shape)
       
    return qs_all

# wrapper:
#
def quantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, interpolation=None):
    dict_methods = {'inverted_cdf':inverted_cdf,
                    'averaged_inverted_cdf':averaged_inverted_cdf,
                    'closest_observation':closest_observation,
                    'interpolated_inverted_cdf':interpolated_inverted_cdf,
                    'hazen':hazen,
                    'weibull':weibull,
                    'linear':linear,
                    'median_unbiased':median_unbiased,
                    'normal_unbiased':normal_unbiased,
                    'lower':lower,
                    'higher':higher,
                    'midpoint':midpoint,
                    'nearest':nearest}

    if axis != None and (not cu.isscalar(axis)):
        if len(axis) == 1:
            real_axis = axis[0]
            a_rr = a
        else:
            (real_axis, a_rr) = reshuffle_reshape(a, axis)
            # What happens with multiple axes and overwrite_input = True ?
            # It seems overwrite_input is reset to False;
            overwrite_input = False
    else:
        real_axis = axis
        a_rr = a
    

    if cu.isscalar(q):
        q_arr = cu.array([q])
        # TODO: the result must also be reduced in dimension, accordingly;
    else:
        q_arr = q
    
    # in the future k-sort (partition)
    # might be faster, for now it uses sort
    # arr = partition(arr, k = floor(nq), axis = real_axis)
    # but that would require a k-sort call for each `q`!
    # too expensive for many `q` values...
    # if no axis given then elements are sorted as a 1D array
    #
    if overwrite_input:
        a_rr.sort(axis = real_axis)
        arr = a_rr
    else:
        arr = cu.sort(a_rr, axis = real_axis)

    # return type dependency on arr.dtype:
    #
    # it depends on interpolation method;
    # For discontinuous methods returning either end of the interval within
    # which the quantile falls, or the other; arr.dtype is returned;
    # else, logic below:
    #
    # if is_float(arr_dtype) && (arr.dtype >= dtype('float64')) then
    #    arr.dtype
    # else
    #    dtype('float64')
    #
    # see https://github.com/numpy/numpy/issues/22323
    #
    if method in ['inverted_cdf', 'closest_observation', 'lower', 'higher', 'nearest']:
        to_dtype = arr.dtype
    else:
        to_dtype = arr.dtype if (arr.dtype == cu.dtype('float128')) else cu.dtype('float64')
    
    res = quantile_impl(arr, q_arr, real_axis, dict_methods[method], keepdims, to_dtype)

    if out != None:
        out = res.astype(out.dtype) # conversion from res.dtype to out.dtype
        return out
    else:
        if cu.isscalar(q): # q_arr is singleton from scalar; additional dimension 1 must be removed
            return res[0]
        else:
            return res
