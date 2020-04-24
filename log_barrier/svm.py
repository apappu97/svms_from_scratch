"""
@author: aneesh pappu

This file contains code for training an SVM from scratch, i.e. it formulates
the objective function and solves the quadratic programming convex optimization
problem via a log barrier interior point method with feasible start Newton. The
dual formulation is used in order to permit kernelisation for higher
dimensional representations.

In particular, it contains classes for:
    a) Newton's method (given a function and its derivatives, perform the
    backtracking line search. Essentially, the inner loop iteration)
    b) Barrier (The outerloop iterations of adjusting t to setup the inner
    loop iteration problem)
    c) Objective (Encodes the derivatives for the function i.e. the SVM
    dual objective)
    d) SVM Classifier (class for predicting once Lagrange multiplier dual
    variables have been solved for)

TODO: Separate these classes into their own files
"""
import numpy as np
import scipy

class SVMObjective:
    """
    Implements functions and derivatives for SVM Objective. Note that the
    function is the barrier modified function, not the original dual SVM
    objective
    """
    def __init__(self, X_train, y_train, kernel_func, C):
       self.K = kernel_func(X_train, X_train)
       self.Y = np.diag(y_train)
       self.C = C

    def f(self, a, t):
        """
        Evaluate function f
        """
        assert a.shape == (self.Y.shape[0], 1)
        first =  .5 * a.T @ self.Y @ self.K @ self.Y @ a
        second = np.sum(a)
        third = np.sum(np.log(a))
        fourth = np.sum(np.log(self.C - a))
        return t * (first - second) - third - fourth

    def grad(self, a, t):
        """
        Returns gradient of SVM barrier function
        """
        first = t * (self.Y @ self.K @ self.Y @ a  - np.ones_like(a))
        second = np.array(1.0/a)
        third = np.array(1.0/(self.C - a))

        return first - second + third

    def hessian(self, a, t):
        """
        Returns hessian of SVM barrier function
        """
        first = t * self.Y @ self.K @ self.Y
        second = np.diag(1/a**2)
        third = np.diag(1/(self.C - a)**2)
        return first + second + third

class Newton:
    def __init__(self, curr_a, objective, C, c_1=0.01, rho=0.5):
        """
        Takes in current iterate curr_a, objective function, (c_1, rho)
        parameters for backtracking linesearch
        """
        self.curr_a = curr_a
        self.objective = objective
        self.c_1 =c_1
        self.rho = rho
        self.A = np.vstack((np.eye(curr_a.shape[0]),
            -1*np.eye(curr_a.shape[0])))
        self.C = C

    def newton_step(self, t):
        """
        Solves the KKT system to find the Newton step
        """
        hess = self.objective.hessian(self.curr_a, t)
        grad = self.objective.grad(self.curr_a, t)
        KKT_mat = np.block([[hess, self.A.T], [self.A,
            np.zeros((self.A.shape[0], self.A.T.shape[1]))]])
        rhs = -1 * np.vstack((grad, np.zeros((self.A.shape[0], 1))))

        # The KKT matrix might be singular. This is okay, as long as we are in
        # the case of non-empty null space -- any solution will be dual
        # feasible, so we just use the pseudoinverse to calculate the minimum norm
        # solution.
        try:
            solved = scipy.linalg.solve(KKT_mat, rhs)
        except np.linalg.LinAlgError:
            solved = scipy.linalg.pinv(KKT_mat) @ rhs
        delta_a = solved[:hess.shape[0]] # grab just delta x part
        return delta_a

    def line_search(self, delta_x, t):
        """
        Perform line search on Newton direction from delta_x
        Backtracks until sufficient decrease (Armijo) cond is satisfied and the inequality
        constraints are also satisfied

        Returns step size
        """

        ls_coeff = 1
        # Inequality constraint condition
        while (min(self.curr_a + ls_coeff * delta_x) < 0 or max(self.curr_a +
            ls_coeff * delta_x) > self.C):
            ls_coeff = self.rho * ls_coeff

        # Sufficient decrease conditions
        while True:
            lower_bound = self.objective.f(self.curr_a, t) + self.c_1 * ls_coeff * np.dot(self.objective.grad(self.curr_a, t).T, delta_x)
            if self.objective.f(self.curr_a + ls_coeff * delta_x, t) <= lower_bound:
                break
            ls_coeff = self.rho * ls_coeff

        return ls_coeff

class Barrier:
    """
    Implements Barrier method (i.e. outer loop logic)
    """
    def __init__(self, t_0=1, mu=3, tol=1e-10, max_iter=200):
        self.t_0 = t_0
        self.mu = mu # expansion constant
        self.tol = tol
        self.max_iter = max_iter

    def run(self, X_train, y_train, kernel_func, C, m):
        """
        Runs Barrier method

        Parameters:
        X_train: data as rows
        y_train: array of floats representing labels, +1 and -1
        kernel_func: callable kernel function
        C: SVM C parameter
        m: number of inequality constraints
        """
        a = self._feasible_starting_point(y_train, C)
        print("Feasible starting point: {}".format(a))
        svm_objective = SVMObjective(X_train, y_train, kernel_func, C)
        t = self.t_0
        curr_iter = 0
        while True:
            newton = Newton(a, svm_objective, C)
            unscaled_da = newton.newton_step(t)
            step_length = newton.line_search(unscaled_da, t)
            a_next = a + step_length * unscaled_da
            a = a_next
            # check stopping criteria
            if m/t < self.tol:
                break
            # increase t
            t = t * self.mu
            print("Finished iter: {}".format(curr_iter))
            curr_iter+=1
        print('Final a: {}'.format(a))
        return a

    def _feasible_starting_point(self, y_labels, C):
        """
        Find a strictly feasible starting point for the barrier method
        Thanks to my friend and colleague @Callum Lau for helping me with the
        feasibilty condition
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

class SVMClassifier:
    """
    Takes solved Lagrange multipliers barrier method and creates SVM classifier
    """

    def __init__(self, a, X_train, y_train, kernel_func):
        self.alphas = a
        if len(y_train.shape) != 2 or y_train.shape[1] != 1: y_train = y_train[:,
                None]
        alpha_y = self.alphas * y_train
        if len(alpha_y.shape) != 2 or alpha_y.shape[1] != 1: alpha_y = alpha_y[:, None]
        self.alpha_y = alpha_y
        # calculate b
        K_train = kernel_func(X_train, X_train)
        support_vectors = np.where(self.alphas != 0)[0]
        differences = []
        for ind in support_vectors:
            kernels = K_train[ind][:, None] # kernel value of support vector w train pts
            pred = kernels.T @ self.alpha_y
            diff = np.abs(y_train[ind] - pred)
            differences.append(diff)
        # set b to the median of this differences array
        self.b = np.median(np.array(differences))
        self.kernel_func = kernel_func
        self.X_train = X_train

    def accuracy(self, X_test, y_test, b = None):
        # import pdb; pdb.set_trace()
        kernel_mat = self.kernel_func(X_test, self.X_train)
        print("shape of X_test {} and shape of kernel mat {}, rows should match".format(X_test.shape, kernel_mat.shape))
        if b is None:
            b = self.b
        pred = np.sign(kernel_mat @ self.alpha_y + b)
        pred = np.squeeze(pred)
        print("pred {}, {}".format(pred.shape, pred))
        num_correct = sum(pred == y_test)
        print("num correct {}, accuracy {}".format(num_correct,
            num_correct/len(y_test)))

        return num_correct/len(y_test)
