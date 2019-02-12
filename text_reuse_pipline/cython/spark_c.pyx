# cython: language_level=3

cimport numpy as np
import numpy as np

cdef float cython_cosine(np.ndarray[int, ndim=1] x_i, np.ndarray[double, ndim=1] x_v, np.ndarray[int, ndim=1] y_i, np.ndarray[double, ndim=1] y_v,double a_norm,double b_norm):
    cdef int n_x = x_i.shape[0]
    cdef int n_y = y_i.shape[0]
    
    cdef double a_n = a_norm
    cdef double b_n = b_norm

    cdef float sim = 0.0
    cdef int i = 0
    cdef int j = 0
    while (i<n_x and j<n_y):
        if (x_i[i] == y_i[j]):
            sim += x_v[i]*y_v[j]
            i+=1
            j+=1
        elif (x_i[i]<y_i[j]):
            i+=1
        else:
            j+=1
    return sim/(a_n*b_n)


cpdef cosine_mapper(arr_a, arr_b, threshold):
    
    calc_arr = []
    for entry in arr_a:
        for b_entry in arr_b:        
            if (b_entry[0]<entry[0]) and (b_entry[1][1]!=entry[1][1]):
                similarity = cython_cosine(np.asarray(b_entry[1][0].indices), np.asarray(b_entry[1][0].values), np.asarray(entry[1][0].indices), np.asarray(entry[1][0].values), b_entry[1][2], entry[1][2])
                if (similarity>threshold):
                    calc_arr.append((str(b_entry[1][1])+"-"+str(entry[1][1]), similarity))
                    
    return calc_arr
