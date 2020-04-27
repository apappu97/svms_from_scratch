"""
@author: aneesh pappu
Code for RBF and polynomial kernels
"""
import numpy as np
def kernel_rbf(gamma):
    """
    Calculates gaussian kernel. Expects two matrices, convention of each
    row is a datapoint.
    Returns matrix where row of matrix is the kernels of the datapoint
    corresponding to that index in the x1 matrix with every datapoint
    in the x2 matrix.
    """
    def _kernel_rbf(x1, x2):
        if len(x1.shape) == 1:
            x1 = x1[None,:]
        if len(x2.shape) == 1:
            x2 = x2[None, :]
        x_input_norms = np.diag(x1 @ x1.T)[:, None]
        x_out_norms = np.diag(x2 @ x2.T)[None, :]

        numerator = x_input_norms + x_out_norms
        numerator -= 2*(x1 @ x2.T)
        ret = np.exp(-gamma * numerator)

        return ret
    return _kernel_rbf

def kernel_poly(d=3):
    """
    Same convention as above. If one datapoint is passed in, it will be made
    into a row vector. Polynomial kernel calculates K(x, z) = (x^Tz)^d.
    """
    def _kernel_poly(x1, x2):

        if len(x1.shape) == 1:
            x1 = x1[None,:]
        if len(x2.shape) == 1:
            x2 = x2[None, :]
        K = (x1 @ x2.T)**d
        return K
    return _kernel_poly

def kernel_lin():
    def _kernel_linear(x1, x2):
        if len(x1.shape) == 1:
            x1 = x1[None, :]
        if len(x2.shape) == 1:
            x2 = x2[None, :]
        K = x1 @ x2.T
        return K
    return _kernel_linear
