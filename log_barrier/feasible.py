"""
Code for generating a feasible starting point for optimizing the SVM dual
objective via feasible start Newton + log barrier interior point method
@author: aneesh pappu
thanks to my friend and colleague @Callum Lau for help with this method
"""
import numpy as np
def feasible_starting_point(y_labels, C):
    """
    Find a strictly feasible starting point for the barrier method
    """
    total_num = len(y_labels)
    pos = np.sum(y_labels == 1)
    pos_frac = pos / (total_num - pos)
    a0 = np.zeros((y_labels.shape[0], 1))
    for i in range(a0.shape[0]):
        if y_labels[i] == 1:
            a0[i] = C * (1 - pos/total_num)
        else:
            a0[i] = C * pos_frac * (1 - pos/total_num)
    return a0
