from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta
import numdifftools as nd


class EGPDBase(ABC):

    param_names = ()

    def __init__(self, sigma, xi):

        if sigma <= 0:
            raise ValueError("sigma deve ser > 0")

        self.sigma = sigma
        self.xi = xi

    def _Hbar(self, x):

        x = np.asarray(x)

        if abs(self.xi) < 1e-10:
            return np.exp(-x / self.sigma)

        t = 1 + self.xi * x / self.sigma

        out = np.zeros_like(x, dtype=float)
        out[t > 0] = t[t > 0]**(-1/self.xi)

        return out


    def _H(self, x):
        return 1 - self._Hbar(x)


    def _h(self, x):

        x = np.asarray(x)

        if abs(self.xi) < 1e-10:
            return np.exp(-x / self.sigma)

        t = 1 + self.xi * x / self.sigma

        out = np.zeros_like(x, dtype=float)
        out[t > 0] = t[t > 0]**(-1/self.xi - 1)

        return out

    @abstractmethod
    def _G(self, v):
        pass

    @abstractmethod
    def _g(self, v):
        pass

    @abstractmethod
    def _Ginv(self, u):
        pass

    def cdf(self, x):
        H = self._H(x)
        return self._G(H)

    def pdf(self, x):

        H = self._H(x)
        h = self._h(x)
        return (
            self._g(H) * h / self.sigma
        )

    def logpdf(self, x):
        f = self.pdf(x)
        f = np.clip(f, 1e-300, None)
        return np.log(f)

    def ppf(self, u):

        u = np.asarray(u)

        eps = np.finfo(float).eps
        u = np.clip(u, eps, 1 - eps)

        v = self._Ginv(u)

        if abs(self.xi) < 1e-10:

            return -self.sigma * np.log(1 - v)

        return (
            self.sigma / self.xi
        ) * (
            (1 - v)**(-self.xi) - 1
        )
        
    def rvs(self, size=1, random_state=None):

        rng = np.random.default_rng(random_state)

        u = rng.uniform(size=size)

        return self.ppf(u)
        
    @classmethod
    @abstractmethod
    def default_init(cls):
        pass    

    @classmethod
    @abstractmethod
    def default_bounds(cls):
        pass

    @staticmethod
    def _negloglik(params, data, cls):

        try:

            model = cls(*params)

            ll = np.sum(
                model.logpdf(data)
            )

            if not np.isfinite(ll):
                return np.inf

            return -ll

        except Exception:
            return np.inf

    @classmethod
    def fit(
        cls,
        data,
        init=None,
        bounds=None,
        method="L-BFGS-B",
        return_optimizer=False
    ):

        data = np.asarray(data)

        if init is None:
            init = cls.default_init()

        if bounds is None:
            bounds = cls.default_bounds()

        res = minimize(
            cls._negloglik,
            x0=init,
            args=(data, cls),
            method=method,
            bounds=bounds
        )

        fitted = cls(*res.x)

        fitted.data = data
        fitted.optimizer = res

        if return_optimizer:
            return fitted, res

        return fitted
        
    def standard_errors(
        self,
        method="numerical",
        return_cov=True
    ):

        params = np.array(
            [getattr(self, p)
             for p in self.param_names]
        )

        if method == "numerical":

            H = nd.Hessian(
                lambda p:
                self._negloglik(
                    p,
                    self.data,
                    self.__class__
                )
            )(params)

            cov = np.linalg.inv(H)

        elif method == "optimizer":

            cov = np.array(
                self.optimizer.hess_inv.todense()
            )

        else:

            raise ValueError(
                "method inválido"
            )

        se = np.sqrt(np.diag(cov))

        if return_cov:
            return se, cov

        return se

class NaveauModelI(EGPDBase):

    param_names = (
        "kappa",
        "sigma",
        "xi"
    )

    def __init__(
        self,
        kappa,
        sigma,
        xi
    ):

        super().__init__(sigma, xi)

        self.kappa = kappa
        
    def _G(self, v):
        return v**self.kappa

    def _g(self, v):
        return (
            self.kappa
            * v**(self.kappa - 1)
        )

    def _Ginv(self, u):
        return u**(1/self.kappa)

    @classmethod
    def default_init(cls):
        return (1.0, 1.0, 0.1)

    @classmethod
    def default_bounds(cls):
        return (
            (1e-6, None),
            (1e-6, None),
            (-1, 5)
        )

class NaveauModelIV(EGPDBase):

    param_names = (
        "kappa",
        "delta",
        "sigma",
        "xi"
    )
    
    def __init__(
        self,
        kappa,
        delta,
        sigma,
        xi
    ):

        super().__init__(sigma, xi)

        self.kappa = kappa
        self.delta = delta

    def _G(self, v):

        Hbar = 1 - v

        A = (
            1
            - ((1+self.delta)/self.delta)
            * Hbar
            + (1/self.delta)
            * Hbar**(1+self.delta)
        )

        return A**(self.kappa/2)

    def _g(self, v):

        Hbar = 1 - v

        A = (
            1
            - ((1+self.delta)/self.delta)
            * Hbar
            + (1/self.delta)
            * Hbar**(1+self.delta)
        )

        return (
            self.kappa
            * (1+self.delta)
            / (2*self.delta)
            * A**(self.kappa/2 - 1)
            * (1 - Hbar**self.delta)
        )

    def _Ginv(self, u):

        z = beta.ppf(
            1 - u**(2/self.kappa),
            a=1/self.delta,
            b=2
        )

        return (
            1
            - z**(1/self.delta)
        )

    @classmethod
    def default_init(cls):
        return (1.0, 1.0, 1.0, 0.1)

    @classmethod
    def default_bounds(cls):
        return (
            (1e-6, None),  # kappa
            (1e-6, None),  # delta
            (1e-6, None),  # sigma
            (-1, 5)        # xi
        )