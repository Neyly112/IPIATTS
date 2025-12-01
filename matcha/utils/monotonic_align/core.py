"""Pure Python implementation of monotonic alignment (fallback when Cython not available)"""
import numpy as np


def maximum_path_each(path, value, t_x, t_y, max_neg_val=-1e9):
    """Python implementation of maximum_path_each"""
    index = t_x - 1
    
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[x, y-1]
            
            if x == 0:
                if y == 0:
                    v_prev = 0.
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[x-1, y-1]
            
            value[x, y] = max(v_cur, v_prev) + value[x, y]
    
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y-1] < value[index-1, y-1]):
            index = index - 1


def maximum_path_c(paths, values, t_xs, t_ys, max_neg_val=-1e9):
    """Python implementation of maximum_path_c"""
    b = values.shape[0]
    
    for i in range(b):
        maximum_path_each(paths[i], values[i], t_xs[i], t_ys[i], max_neg_val)
