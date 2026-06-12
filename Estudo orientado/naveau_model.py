import numpy as np
from scipy.optimize import minimize
import numdifftools as nd
from scipy.stats import beta


class NaveauModelI:
    """
    Modelo (i) de Naveau et al. (2016)
    F(x) = [H_xi(x/sigma)]^kappa
    onde H_xi é a CDF da GPD.
    """

    def __init__(self, kappa, sigma, xi, xi1=0):

        if kappa <= 0:
            raise ValueError("kappa deve ser > 0")

        if sigma <= 0:
            raise ValueError("sigma deve ser > 0")

        self.kappa = kappa
        self.sigma = sigma
        self.xi = xi
        self.xi1 = xi1

    def _xi_t(self, x, dummy):

        if dummy is None:
            dummy = np.zeros_like(x, dtype=float)  
        dummy = np.asarray(dummy, dtype=float)
        xi_t = self.xi + dummy * self.xi1

        return xi_t

    # =========================================================
    # CDF
    # =========================================================
    def cdf(self, x, dummy=None):

        x = np.asarray(x, dtype=float)
        xi_t = self._xi_t(x, dummy)
        H = np.zeros_like(x, dtype=float)
        # caso xi_t = 0
        mask0 = np.abs(xi_t) < 1e-10
        H[mask0] = 1 - np.exp(-x[mask0] / self.sigma)
        # caso xi_t != 0
        mask1 = ~mask0
        t = 1 + xi_t[mask1] * x[mask1] / self.sigma
        H1 = np.ones_like(t)
        valid = t > 0
        H1[valid] = 1 - t[valid]**(-1 / xi_t[mask1][valid])
        H[mask1] = H1
        return H**self.kappa

    # =========================================================
    # PDF
    # =========================================================
    def pdf(self, x, dummy=None):

        x = np.asarray(x)
        xi_t = self._xi_t(x, dummy)
        H = np.zeros_like(x, dtype=float)
        f = np.zeros_like(x, dtype=float)
        
        # caso xi_t = 0 (exponencial)
        mask0 = np.abs(xi_t) < 1e-10
        exp_term = np.exp(-x[mask0] / self.sigma)
        H[mask0] = 1 - exp_term
        H[mask0] = np.clip(H[mask0], 1e-300, 1)
        f[mask0] = (
            (self.kappa / self.sigma)
            * H[mask0]**(self.kappa - 1)
            * exp_term
        )
        # caso xi_t = 0
        mask1 = ~mask0
        t = 1 + xi_t[mask1] * x[mask1] / self.sigma
        valid = t > 0
        H1, f1 = np.zeros_like(t), np.zeros_like(t)
        H1[valid] = 1 - t[valid]**(-1 / xi_t[mask1][valid])
        H1 = np.clip(H1, 1e-300, 1)
        f1[valid] = (
            (self.kappa / self.sigma)
            * H1[valid]**(self.kappa - 1)
            * t[valid]**(-1 / xi_t[mask1][valid] - 1)
        )
        f[mask1] = f1
        return f

    # =========================================================
    # LOGPDF
    # =========================================================
    def logpdf(self, x, dummy=None):

        x = np.asarray(x)
        xi_t = self._xi_t(x, dummy)
        H = np.zeros_like(x, dtype=float)
        logf = np.full_like(x, -np.inf, dtype=float)

        # caso xi_t = 0 (exponencial)
        mask0 = np.abs(xi_t) < 1e-10
        exp_term = np.exp(-x[mask0] / self.sigma)
        H[mask0] = 1 - exp_term
        H[mask0] = np.clip(H[mask0], 1e-300, 1)
        
        logf[mask0] = (
            np.log(self.kappa)
            - np.log(self.sigma)
            + (self.kappa - 1) * np.log(H[mask0])
            - x[mask0] / self.sigma
        )

        # caso xi_t != 0
        mask1 = ~mask0
        t = 1 + xi_t[mask1] * x[mask1] / self.sigma
        valid = t > 0
        H1 = np.zeros_like(t)
        logf1 = np.full_like(t, -np.inf)
        H1[valid] = 1 - t[valid]**(-1 / xi_t[mask1][valid])
        H1 = np.clip(H1, 1e-300, 1)
        logf1[valid] = (
            np.log(self.kappa)
            - np.log(self.sigma)
            + (self.kappa - 1) * np.log(H1[valid])
            - (1 / xi_t[mask1][valid] + 1) * np.log(t[valid])
        )
        H[mask1] = H1
        logf[mask1] = logf1
        return logf

    # =========================================================
    # PPF (inverse CDF)
    # =========================================================
    def ppf(self, u, dummy=None):
        u = np.asarray(u)
        if np.any((u < 0) | (u > 1)):
            raise ValueError("u deve estar em (0,1)")

        xi_t = self._xi_t(u, dummy)
        x = np.zeros_like(u, dtype=float)

        eps = np.finfo(float).eps
        u = np.clip(u, eps, 1 - eps)
        v = u**(1 / self.kappa)

        # caso exponencial
        mask0 = np.abs(xi_t) < 1e-10
        x[mask0] = -self.sigma * np.log(1 - v[mask0])
        # caso xi != 0
        mask1 = ~mask0
        x[mask1] = (
            self.sigma / xi_t[mask1]
        ) * (
            (1 - v[mask1])**(-xi_t[mask1]) - 1
        )
        return x

    # =========================================================
    # RANDOM SAMPLING
    # =========================================================
    def rvs(self, size=1, random_state=None, dummy=None):
        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=size)
        return self.ppf(u, dummy=dummy)

        
    # =====================================================
    # negative loglikelihood
    # =====================================================
    @staticmethod
    def _negloglik(params, data, dummy, cls):
        
        # restrições de kappa e sigma
        if params[0] <= 0 or params[1] <= 0:
            return np.inf
    
        try:
            model = cls(*params)
            logf = model.logpdf(data, dummy)
            
            if np.any(~np.isfinite(logf)):
                return np.inf
            return -np.sum(logf)
    
        except Exception:
            return np.inf

    @staticmethod
    def _negloglik_censored(params, data, C, dummy, cls):
        
        # restrições de kappa e sigma
        if params[0] <= 0 or params[1] <= 0:
            return np.inf
    
        try:
            model = cls(*params)            
            ll = 0

            # parte densidade
            upper = data >= C
            logf = model.logpdf(
                data[upper], dummy[upper] if dummy is not None else None
            )
            if np.any(~np.isfinite(logf)):
                return np.inf
            ll += np.sum(logf)

            # parte censurada (CDF)
            lower = data < C
            Fc = model.cdf(
                np.full(len(data[lower]), C),
                dummy[lower] if dummy is not None else None
            )
            ll += np.sum(np.log(Fc))
            return -ll
    
        except Exception:
            return np.inf

    @classmethod
    def fit(cls, data, init=(1.0, 1.0, 0.1, 0.0),
        bounds=((1e-6, None), (1e-6, None), (-1, 5), (-5, 5)),
        method="L-BFGS-B", return_optimizer=False,
        censored=False, C=0.5, dummy=None):
    
        data = np.asarray(data)
        if dummy is None:
            init = init[:3]
            bounds = bounds[:3]

        if censored:
            objective = cls._negloglik_censored
            args = (data, C, dummy, cls)
        else:
            objective = cls._negloglik
            args = (data, dummy, cls)
            
        # otimização
        res = minimize(objective, x0=init, args=args, 
                method=method, bounds=bounds
        )
        # objeto ajustado
        fitted_model = cls(*res.x)
        fitted_model.optimizer = res
        fitted_model.data = data
        fitted_model.censored = censored
        fitted_model.C = C
        fitted_model.dummy = dummy

        if return_optimizer:
            return fitted_model, res
        return fitted_model

    # =====================================================
    # STANDARD ERRORS
    # =====================================================
    def standard_errors(self, method="numerical", return_cov=True):

        params = [self.kappa, self.sigma, self.xi]
        if self.dummy is not None: params.append(self.xi1)
        params = np.array(params)

        # Hessiana numérica
        if method == "numerical":
            if self.censored is False:
                H = nd.Hessian(lambda p: self._negloglik(p, self.data, self.dummy, self.__class__))(params) 
            else:
                H = nd.Hessian(lambda p: self._negloglik_censored(p, self.data, self.C, self.dummy, self.__class__))(params) 
            cov = np.linalg.inv(H)
            # cov = np.linalg.pinv(H)

        # Hessiana do optimizer
        elif method == "optimizer":
            if not hasattr(self, "optimizer"):
                raise ValueError(
                    "optimizer não encontrado. "
                    "Use fit() antes."
                )
            cov = np.array(self.optimizer.hess_inv.todense())

        else:
            raise ValueError(
                "method deve ser "
                "'numerical' ou 'optimizer'"
            )

        # erros padrão
        se = np.sqrt(np.diag(cov))
        if return_cov:
            return se, cov
        return se

    # =====================================================
    # SUMMARY
    # =====================================================
    def summary(self, print_results=True):
        
        se_num, _ = self.standard_errors(method="numerical")
        results = []
        
        upper, lower = self.kappa + 1.96*se_num[0], self.kappa - 1.96*se_num[0]
        results.append("{:.3f} ({:.2f}, {:.2f})".format(self.kappa, lower, upper))
        
        upper, lower = self.sigma + 1.96*se_num[1], self.sigma - 1.96*se_num[1]
        results.append("{:.3f} ({:.2f}, {:.2f})".format(self.sigma, lower, upper))
        
        upper, lower = self.xi + 1.96*se_num[2], self.xi - 1.96*se_num[2]
        results.append("{:.3f} ({:.2f}, {:.2f})".format(self.xi, lower, upper))
        
        if self.dummy is not None:
            upper, lower = self.xi1 + 1.96*se_num[3], self.xi1 - 1.96*se_num[3]
            results.append("{:.3f} ({:.2f}, {:.2f})".format(self.xi1, lower, upper))

        if print_results:
            print("kappa = " + results[0])
            print("sigma = " + results[1])
            print("xi = " + results[2])
            if self.dummy is not None:
                print("xi1 = " + results[3])
            
        return results


class NaveauModelIV:
    """
    Modelo (iv) de Naveau et al. (2016)

    G(v) = [1 - Q_delta((1-v)^delta)]^(kappa/2)

    onde Q_delta é a CDF da Beta(1/delta, 2).
    """

    def __init__(self, kappa, delta, sigma, xi):

        if kappa <= 0:
            raise ValueError("kappa deve ser > 0")

        if delta <= 0:
            raise ValueError("delta deve ser > 0")

        if sigma <= 0:
            raise ValueError("sigma deve ser > 0")

        self.kappa = kappa
        self.delta = delta
        self.sigma = sigma
        self.xi = xi

    # =========================================================
    # Survival da GPD
    # =========================================================
    def _Hbar(self, x):

        x = np.asarray(x)

        if np.abs(self.xi) < 1e-10:
            return np.exp(-x / self.sigma)

        t = 1 + self.xi * x / self.sigma

        Hbar = np.zeros_like(x, dtype=float)
        Hbar[t > 0] = t[t > 0]**(-1/self.xi)

        return Hbar

    # =========================================================
    # Densidade da GPD
    # =========================================================
    def _h(self, x):

        x = np.asarray(x)

        if np.abs(self.xi) < 1e-10:
            return np.exp(-x / self.sigma)

        t = 1 + self.xi * x / self.sigma

        h = np.zeros_like(x, dtype=float)
        h[t > 0] = t[t > 0]**(-1/self.xi - 1)

        return h

    # =========================================================
    # CDF
    # =========================================================
    def cdf(self, x):

        Hbar = self._Hbar(x)

        A = (
            1
            - ((1 + self.delta) / self.delta) * Hbar
            + (1 / self.delta) * Hbar**(1 + self.delta)
        )

        A = np.clip(A, 0, 1)

        return A**(self.kappa / 2)

    # =========================================================
    # PDF
    # =========================================================
    def pdf(self, x):

        x = np.asarray(x)

        Hbar = self._Hbar(x)
        h = self._h(x)

        A = (
            1
            - ((1 + self.delta) / self.delta) * Hbar
            + (1 / self.delta) * Hbar**(1 + self.delta)
        )

        A = np.clip(A, 1e-300, None)

        f = (
            self.kappa
            * (1 + self.delta)
            / (2 * self.delta * self.sigma)
            * h
            * A**(self.kappa / 2 - 1)
            * (1 - Hbar**self.delta)
        )

        return f

    # =========================================================
    # LOGPDF
    # =========================================================
    def logpdf(self, x):

        x = np.asarray(x)

        Hbar = self._Hbar(x)
        h = self._h(x)

        A = (
            1
            - ((1 + self.delta) / self.delta) * Hbar
            + (1 / self.delta) * Hbar**(1 + self.delta)
        )

        A = np.clip(A, 1e-300, None)
        h = np.clip(h, 1e-300, None)

        logf = (
            np.log(self.kappa)
            + np.log(1 + self.delta)
            - np.log(2)
            - np.log(self.delta)
            - np.log(self.sigma)
            + (self.kappa / 2 - 1) * np.log(A)
            + np.log(1 - Hbar**self.delta)
            + np.log(h)
        )

        return logf

    # =========================================================
    # PPF
    # =========================================================
    def ppf(self, u):

        u = np.asarray(u)

        eps = np.finfo(float).eps
        u = np.clip(u, eps, 1 - eps)

        # Quantil Beta(1/delta, 2)
        z = beta.ppf(
            1 - u**(2 / self.kappa),
            a=1/self.delta,
            b=2
        )

        Hbar = z**(1/self.delta)

        if np.abs(self.xi) < 1e-10:

            return -self.sigma * np.log(Hbar)

        return (
            self.sigma / self.xi
        ) * (
            Hbar**(-self.xi) - 1
        )

    # =========================================================
    # RANDOM SAMPLING
    # =========================================================
    def rvs(self, size=1, random_state=None):

        rng = np.random.default_rng(random_state)

        u = rng.uniform(size=size)

        return self.ppf(u)
    
        