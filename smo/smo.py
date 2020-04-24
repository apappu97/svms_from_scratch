"""
@author: aneesh pappu

This file contains code for training an SVM from scratch using the Sequential
Minimal Optimization (SMO) algorithm

Pseudocode found at: http://cs229.stanford.edu/materials/smo.pdf was used as
reference
"""

import numpy as np
import scipy


class SMO:
    """
    Implements logic for SMO Optimizer
    """
    def __init__(self, X_train, y_train, kernel_func, C, tol=1e-10):
        self.X_train = X_train
        if len(y_train.shape) == 1 :
            y_train = y_train[:, None]
        self.y_train = y_train
        self.kernel_func = kernel_func
        self.C = C
        self.tol = tol
        self.K = kernel_func(X_train, X_train)
        self.Y = np.diag(np.squeeze(y_train))
        self.alphas = np.zeros((self.y_train.shape[0], 1))
        self.b = 0

    def _calculate_alphay(self):
        """
        Simple helper for calculating alpha * y element wise, needed for
        reconstructing the weight vector/making predictions
        """
        assert self.alphas.shape[1] == 1 and len(self.alphas.shape) == 2
        assert self.y_train.shape[1] == 1 and len(self.y_train.shape) == 2
        alpha_y = self.alphas * self.y_train
        assert alpha_y.shape[1] == 1
        return alpha_y

    def lagrange_dual_objective(self):
        """
        Evaluates Lagrange dual objective at current iterates
        """
        assert self.alphas.shape == (self.Y.shape[0], 1)
        first = np.sum(self.alphas)
        second =  .5 * self.alphas.T @ self.Y @ self.K @ self.Y @ self.alphas
        return first - second

    def compute_fx(self, x_test):
        """
        Computes f(x) = \sum_i alpha_iy^iK(x_i, x_test) + b
        """
        # import pdb; pdb.set_trace()
        kernel_values = self.kernel_func(x_test, self.X_train)
        alpha_y = self._calculate_alphay()
        fx = np.dot(kernel_values, alpha_y) + self.b
        return fx

    def calculate_E(self, idx):
        fx = self.compute_fx(self.X_train[idx, :][None, :])
        y = self.y_train[idx]
        E= fx - y
        # import pdb; pdb.set_trace()
        E = np.asscalar(E)
        return E

    def calculate_eta(self, i, j):
        x_i = self.X_train[i, :][None, :]
        x_j = self.X_train[j, :][None, :]
        eta = 2 * self.kernel_func(x_i, x_j) - self.kernel_func(x_i, x_i) - self.kernel_func(x_j, x_j)
        return eta

    def calculate_lower(self, i, j):
        """
        Computes lower bound for alpha_j
        """
        if self.y_train[i] != self.y_train[j]:
            L = max(0, self.alphas[j] - self.alphas[i])
        else:
            L = max(0, self.alphas[i] + self.alphas[j] - self.C)
        return L

    def calculate_upper(self, i, j):
        """
        Computes upper bound for alpha_j
        """
        if self.y_train[i] != self.y_train[j]:
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        else:
            H = min(self.C, self.alphas[i] + self.alphas[j])
        return H

    def clip_alpha(self, alpha, L, H):
        """
        Returned value of clipped alpha based on L (lower) and H (upper) bounds
        """
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha

    def calculate_b1(self, i, j, alpha_i_old, alpha_i, alpha_j_old, alpha_j):
        E_i = self.calculate_E(i)
        third = self.y_train[i] * (alpha_i - alpha_i_old) * self.kernel_func(self.X_train[i, :][None, :], self.X_train[i, :][None,:])
        fourth = self.y_train[j] * (alpha_j - alpha_j_old) * self.kernel_func(self.X_train[i, :][None, :], self.X_train[j, :][None,:])
        b1 = self.b - E_i - third - fourth
        return b1

    def calculate_b2(self, i, j, alpha_i_old, alpha_i, alpha_j_old, alpha_j):
        E_j = self.calculate_E(j)
        third = self.y_train[i] * (alpha_i - alpha_i_old) * self.kernel_func(self.X_train[i, :][None, :], self.X_train[j, :][None,:])
        fourth = self.y_train[j] * (alpha_j - alpha_j_old) * self.kernel_func(self.X_train[j, :][None, :], self.X_train[j, :][None,:])
        b2 = self.b - E_j - third - fourth
        return b2

    def calculate_final_b(self, b1, b2, alpha_i, alpha_j):
        if alpha_i > 0 and self.C > alpha_i:
            return b1
        elif alpha_j > 0 and self.C > alpha_j:
            return b2
        else:
            return (b1 + b2)/2

    def run(self, max_passes=50):
        passes = 0
        while passes < max_passes:
            num_changed_alphas = 0
            #really simple heuristic, pick do scan over each alpha and pick
            # alphas randomly
            for i in range(len(self.alphas)):
                E_i = self.calculate_E(i)
                if (self.y_train[i]*E_i < -1*self.tol and self.alphas[i] < self.C) or (self.y_train[i] * E_i > self.tol and self.alphas[i] > 0):
                    j = np.random.randint(0, len(self.alphas))
                    while j == i:
                        j = np.random.randint(0, len(self.alphas))
                    E_j = self.calculate_E(j)
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    L = self.calculate_lower(i, j)
                    H = self.calculate_upper(i, j)
                    if L == H: continue
                    eta = self.calculate_eta(i, j)
                    if eta >= 0: continue
                    alpha_j = alpha_j_old - self.y_train[j]*(E_i - E_j)/eta
                    alpha_j = self.clip_alpha(alpha_j, L, H)
                    if np.abs(alpha_j - alpha_j_old) < 1e-5: continue
                    alpha_i = alpha_i_old + self.y_train[i]*self.y_train[j]*(alpha_j_old - alpha_j)
                    b1 = self.calculate_b1(i, j, alpha_i_old, alpha_i, alpha_j_old, alpha_j)
                    b2 = self.calculate_b2(i, j, alpha_i_old, alpha_i, alpha_j_old, alpha_j)
                    b = self.calculate_final_b(b1, b2, alpha_i, alpha_j)

                    self.alphas[i] = alpha_i
                    self.alphas[j] = alpha_j
                    num_changed_alphas +=1
            if num_changed_alphas == 0: passes +=1
            else: passes = 0
        return self.alphas, b
