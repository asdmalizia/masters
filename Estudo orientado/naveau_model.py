import numpy as np


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

        if np.any((u <= 0) | (u >= 1)):
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