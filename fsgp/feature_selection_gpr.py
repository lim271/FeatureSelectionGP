import numpy as np
from operator import itemgetter
from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.utils.optimize import _check_optimize_result
from .kernels import RBF, WhiteKernel, ConstantKernel as C



class FeatureSelectionGPR(GaussianProcessRegressor):
    """
    FeatureSelectionGPR - Gaussian process regression with $l_1$-regularization
    """

    def __init__(self, kernel=None, regularization_l1=1.0, regularization_l2=1e-2, regularization_noise=1e3,
            *, alpha=1e-10, rho=1.0, n_restarts_optimizer=0, admm_maxiter=100, admm_tol=5e-3,
            normalize_y=False, copy_X_train=True, random_state=None):
        self.kernel = kernel
        self.regularization_l1 = regularization_l1
        self.regularization_l2 = regularization_l2
        self.regularization_noise = regularization_noise
        self.alpha = alpha
        self.rho = rho
        self.optimizer = "fmin_l_bfgs_b"
        self.n_restarts_optimizer = n_restarts_optimizer
        self.admm_maxiter = admm_maxiter
        self.admm_tol = admm_tol
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state


    def fit(self, X, y):
        """Fit Gaussian process regression model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values
        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(constant_value=1.0, constant_value_bounds=(1e-5, 1e5)) \
                * RBF(
                    length_scale=np.ones((X.shape[1])),
                    length_scale_bounds=(0, 1e5)
                ) \
                + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))
        else:
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            X, y = self._validate_data(X, y, multi_output=True, y_numeric=True,
                                       ensure_2d=True, dtype="numeric")
        else:
            X, y = self._validate_data(X, y, multi_output=True, y_numeric=True,
                                       ensure_2d=False, dtype=None)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = np.std(y, axis=0)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std

        else:
            self._y_train_mean = np.zeros(1)
            self._y_train_std = 1

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def shrinkage(x, kappa):
                return np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)

            # First optimize starting from theta specified in kernel
            def ADMM(initial_theta, bounds):
                _theta = initial_theta
                w = _theta[1:-1]
                w_old = w
                u = np.zeros_like(w)
                _abstol = len(w) * self.admm_tol ** 2
                for _iter in range(self.admm_maxiter):
                    def obj_func(theta, eval_gradient=True):
                        _reg  = self.regularization_l1 * np.sum(np.abs(w))
                        _reg += self.regularization_l2 * np.dot(theta[1:-1], theta[1:-1]) * .5
                        _reg += self.regularization_noise * theta[1:-1] ** 2 * .5
                        _res = theta[1:-1] - w
                        _aug_dual = self.rho * np.dot(u + _res / 2, _res)
                        if eval_gradient:
                            lml, grad = self.log_marginal_likelihood(
                                theta, eval_gradient=True, clone_kernel=False
                            )
                            grad[1:-1] -= self.rho * (_res + u)
                            grad[1:-1] -= self.regularization_l2 * theta[1:-1]
                            grad[-1]   -= self.regularization_noise * theta[-1]
                            return _reg + _aug_dual - lml, -grad
                        else:
                            return _reg + _aug_dual - self.log_marginal_likelihood(
                                theta, clone_kernel=False
                            )
                    sol = self._constrained_optimization(
                        obj_func,
                        _theta,
                        bounds
                    )
                    _theta = sol[0]
                    w  = shrinkage(_theta[1:-1] + u, self.regularization_l1 / self.rho)
                    u += (_theta[1:-1] - w)
                    if (np.linalg.norm(_theta[1:-1] - w) < _abstol) and \
                        (np.linalg.norm(w - w_old) < _abstol + self.admm_tol * np.linalg.norm(self.rho*u)):
                        print('ADMM stopped after ', _iter+1, 'iterations.')
                        break
                    w_old = w
                return sol

            optima = [
                (
                    ADMM(self.kernel_.theta, self.kernel_.bounds)
                )
            ]

            # Additional runs are performed from log-uniform chosen initial theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                while self.n_restarts_optimizer > len(optima)-2:
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        ADMM(
                            theta_initial, bounds
                        )
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta,
                                             clone_kernel=False)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        return self
