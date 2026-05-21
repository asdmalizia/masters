import numpy as np
from scipy.optimize import minimize
import numdifftools as nd


class NaveauModelI:
    """
    Modelo (i) de Naveau et al. (2016)
    F(x) = [H_xi(x/sigma)]^kappa
    onde H_xi é a CDF da GPD.
    """

    def __init__(self, kappa, sigma, xi):

        if kappa <= 0:
            raise ValueError("kappa deve ser > 0")

        if sigma <= 0:
            raise ValueError("sigma deve ser > 0")

        self.kappa = kappa
        self.sigma = sigma
        self.xi = xi

    # =========================================================
    # CDF
    # =========================================================
    def cdf(self, x):

        x = np.asarray(x)
        if np.abs(self.xi) < 1e-10:
            H = 1 - np.exp(-x / self.sigma)

        else:
            t = 1 + self.xi * x / self.sigma
            t = np.maximum(t, 0)
            H = 1 - t**(-1/self.xi)

        return H**self.kappa

    # =========================================================
    # PDF
    # =========================================================
    def pdf(self, x):
        x = np.asarray(x)
        
        # caso exponencial
        if np.abs(self.xi) < 1e-10:
            exp_term = np.exp(-x / self.sigma)
            H = 1 - exp_term
            H = np.clip(H, 1e-300, 1)
            f = (
                (self.kappa / self.sigma)
                * H**(self.kappa - 1)
                * exp_term
            )
            return f

        # caso xi != 0
        else:
            t = 1 + self.xi * x / self.sigma
            H = 1 - t[t>0]**(-1 / self.xi)
            H = np.clip(H, 1e-300, 1)

            f = np.zeros_like(x, dtype=float)
            f[t>0] = (
                (self.kappa / self.sigma)
                * H**(self.kappa - 1)
                * t[t>0]**(-1 / self.xi - 1)
            )
            return f

    # =========================================================
    # LOGPDF
    # =========================================================
    def logpdf(self, x):

        x = np.asarray(x)
        if np.abs(self.xi) < 1e-10:

            exp_term = np.exp(-x / self.sigma)
            H = 1 - exp_term
            H = np.clip(H, 1e-300, 1)
            
            logf = (
                np.log(self.kappa)
                - np.log(self.sigma)
                + (self.kappa - 1) * np.log(H)
                - x / self.sigma
            )
            return logf

        else:
            t = 1 + self.xi * x / self.sigma
            t = np.maximum(t, 0)
            H = 1 - t**(-1 / self.xi)
            H = np.clip(H, 1e-300, 1)
            
            logf = np.full_like(x, -np.inf, dtype=float)
            logf[t>0] = (
                np.log(self.kappa)
                - np.log(self.sigma)
                + (self.kappa - 1) * np.log(H[t>0])
                - (1 / self.xi + 1) * np.log(t[t>0])
            )
            return logf

    # =========================================================
    # PPF (inverse CDF)
    # =========================================================
    def ppf(self, u):
        u = np.asarray(u)
        if np.any((u < 0) | (u > 1)):
            raise ValueError("u deve estar em (0,1)")

        eps = np.finfo(float).eps
        u = np.clip(u, eps, 1 - eps)
        v = u**(1 / self.kappa)
        # caso exponencial
        if np.abs(self.xi) < 1e-10:
            x = -self.sigma * np.log(1 - v)
            return x
        # caso xi != 0
        else:
            x = (
                self.sigma / self.xi
            ) * (
                (1 - v)**(-self.xi) - 1
            )
            return x

    # =========================================================
    # RANDOM SAMPLING
    # =========================================================
    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=size)
        return self.ppf(u)

        
    # =====================================================
    # negative loglikelihood
    # =====================================================
    @staticmethod
    def _negloglik(params, data, cls):
        
        kappa, sigma, xi = params
        if kappa <= 0 or sigma <= 0: # restrições
            return np.inf
    
        try:
            model = cls(kappa, sigma, xi)
            logf = model.logpdf(data)
            
            if np.any(~np.isfinite(logf)):
                return np.inf
            return -np.sum(logf)
    
        except Exception:
            return np.inf

    @staticmethod
    def _negloglik_censored(params, data, C, cls):
        
        kappa, sigma, xi = params
        if kappa <= 0 or sigma <= 0: # restrições
            return np.inf
    
        try:
            model = cls(kappa, sigma, xi)
            
            ll = 0
            # parte densidade
            logf = model.logpdf(data[data>=C])
            if np.any(~np.isfinite(logf)):
                return np.inf
            ll += np.sum(logf)
            # parte censurada (CDF)
            ll += np.log(model.cdf(C)) * len(data[data < C])
            return -ll
    
        except Exception:
            return np.inf

    @classmethod
    def fit(cls, data, init=(1.0, 1.0, 0.1),
        bounds=((1e-6, None), (1e-6, None), (-1, 5)),
        method="L-BFGS-B", return_optimizer=False,
        censored=False, C=0.5):
    
        data = np.asarray(data)
        if censored:
            objective = cls._negloglik_censored
            args = (data, C, cls)
        else:
            objective = cls._negloglik
            args = (data, cls)
            
        # otimização
        res = minimize(
            objective,
            x0=init,
            args=args,
            method=method,
            bounds=bounds
        )
        # objeto ajustado
        fitted_model = cls(*res.x)
        fitted_model.optimizer = res
        fitted_model.data = data
        fitted_model.censored = censored
        fitted_model.C = C
    
        if return_optimizer:
            return fitted_model, res
        return fitted_model

    # =====================================================
    # STANDARD ERRORS
    # =====================================================
    def standard_errors(self, method="numerical", return_cov=True):

        params = np.array([self.kappa, self.sigma, self.xi])

        # Hessiana numérica
        if method == "numerical":
            H = nd.Hessian(lambda p: self._negloglik(p, self.data, self.__class__))(params) 
            cov = np.linalg.inv(H)

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
    def summary(self):

        se_num, _ = self.standard_errors(
            method="numerical"
        )
        se_opt, _ = self.standard_errors(
            method="optimizer"
        )

        # =================================================
        # impressão
        # =================================================
        # print("\n===================================")
        # print("Naveau Model I - Fit Summary")
        # print("===================================\n")
        print("kappa = {:.2f} (SE num = {:.2f}, SE opt = {:.2f})".format(
                self.kappa, se_num[0], se_opt[0])
        )
        print("sigma = {:.2f} (SE num = {:.2f}, SE opt = {:.2f})".format(
            self.sigma, se_num[1], se_opt[1]) 
        )
        print("xi = {:.2f} (SE num = {:.2f}, SE opt = {:.2f})".format(
            self.xi, se_num[2], se_opt[2])
        )
        # print("\n===================================")
    
        