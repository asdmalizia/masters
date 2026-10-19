import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import gamma, expit
from scipy.stats import genextreme


'''
HarveyGEV
│
├── _standardized_residual()
│
├── pdf()
├── logpdf()
│
├── score() -> não padronizazdo
├── fisher_information()
│
├── score_derivative() -> padronizado
├── find_c()
├── find_d()
├── composite_score() -> padronizado
│
├── filter()
├── simulate()
│
├── loglikelihood()│
└── fit()
'''

def _standardized_residual(y, mu, gamma, varphi):
    """
    Calcula o resíduo padronizado:

        x_t = (y_t - mu_{t|t-1}) / varphi

    Parameters
    ----------
    y : float or np.ndarray
        Observação(ões) y_t.
    mu : float or np.ndarray
        Localização condicional mu_{t|t-1}.
    varphi : float
        Parâmetro de escala, varphi > 0.

    Returns
    -------
    x : float or np.ndarray
        Resíduo padronizado x_t.
    """
    return (y - mu - gamma) / varphi

def pdf(y, mu, varphi, xi, alpha, beta):
    """
    Densidade GEV condicional.

    Parameters
    ----------
    y : float or np.ndarray
        Observação(ões) y_t.
    mu : float or np.ndarray
        Localização condicional mu_{t|t-1}.
    varphi : float
        Parâmetro de escala, varphi > 0.
    xi : float
        Parâmetro de forma.

    Returns
    -------
    p : float or np.ndarray
        Densidade condicional.
    """
    t = np.arange(1, len(y) + 1)
    gamma_t = seasonal_component(t, alpha, beta)
    x = _standardized_residual(y, mu, gamma, varphi)

    if xi != 0:
        z = 1 + xi * x

        if np.any(z <= 0):
            return 0.0

        return (
            (1 / varphi)
            * z ** (-1 - 1 / xi)
            * np.exp(-z ** (-1 / xi))
        )

    else:
        return (
            (1 / varphi)
            * np.exp(-x)
            * np.exp(-np.exp(-x))
        )

def logpdf(y, mu, varphi, xi):
    """
    Log-densidade GEV condicional.

    Parameters
    ----------
    y : float or np.ndarray
        Observação(ões) y_t.
    mu : float or np.ndarray
        Localização condicional mu_{t|t-1}.
    varphi : float
        Parâmetro de escala, varphi > 0.
    xi : float
        Parâmetro de forma.

    Returns
    -------
    log_p : float or np.ndarray
        Log-densidade condicional.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    x = _standardized_residual(y, mu, varphi)

    if xi != 0:
        z = 1 + xi * x

        log_p = np.full_like(x, -np.inf, dtype=float)
        mask = z > 0

        log_p[mask] = (
            -np.log(varphi)
            - (1 + 1 / xi) * np.log(z[mask])
            - z[mask] ** (-1 / xi)
        )
        return log_p

    else:
        return (
            -np.log(varphi)
            - x
            - np.exp(-x)
        )

def score(y, xi, mu=0, varphi=1):
    """
    Score da log-densidade em relação a mu_{t|t-1}.

    Parameters
    ----------
    y : float or np.ndarray
        Observação(ões) y_t.
    mu : float or np.ndarray
        Localização condicional mu_{t|t-1}.
    varphi : float
        Parâmetro de escala, varphi > 0.
    xi : float
        Parâmetro de forma.

    Returns
    -------
    score : float or np.ndarray
        Score em relação a mu_{t|t-1}.
    """
    x = _standardized_residual(y, mu, varphi)

    if xi != 0:
        z = 1 + xi * x

        if np.any(z <= 0):
            return np.nan

        return (
            1 / varphi
            * (
                (1 + xi) / z
                - z ** (-1 / xi - 1)
            )
        )

    else:
        return (
            1 / varphi
            * (1 - np.exp(-x))
        )

def fisher_information(xi, varphi=1):
    """
    Informação de Fisher em relação a mu_{t|t-1}.

    Parameters
    ----------
    varphi : float
        Parâmetro de escala, varphi > 0.
    xi : float
        Parâmetro de forma.

    Returns
    -------
    I : float
        Informação de Fisher para mu_{t|t-1}.
    """
    if xi != 0:
        return (
            (1 + xi) ** 2
            * gamma(1 + 2 * xi)
            / varphi ** 2
        )

    else:
        return 1 / varphi ** 2

def score_derivative(x, xi):
    """
    Derivada do score padronizado em relação a x.

    Parameters
    ----------
    x : float or np.ndarray
        Resíduo padronizado x_t.
    xi : float
        Parâmetro de forma, xi > 0.

    Returns
    -------
    derivative : float or np.ndarray
        Derivada s'(x_t).
    """
    z = 1 + xi * x

    if np.any(z <= 0):
        return np.nan

    return (
        -1 / (
            (1 + xi) * gamma(1 + 2 * xi)
        )
        * (
            xi / z**2 - 1 / z**(2 + 1 / xi)
        )
    )

# def find_c(xi):
#     """
#     Encontra c tal que s'(c) = -1.

#     Parameters
#     ----------
#     xi : float
#         Parâmetro de forma, xi > 0.

#     Returns
#     -------
#     c : float
#         Ponto de junção da região linear inferior.
#     """
#     if xi <= 0:
#         raise ValueError("find_c requer xi > 0.")

#     def objective(x):
#         return score_derivative(x, xi) - 1

#     lower, upper = -1 / xi + 1e-10, 1e-10
#     return brentq(objective, lower, upper)

def find_c(xi):
    """
    Encontra c tal que s'(c) = 1.
    Para 0 < xi < 1, a raiz está em aproximadamente
    [-0.5, 0].
    """
    if not 0 < xi < 1:
        raise ValueError("find_c requer 0 < xi < 1.")

    def objective(x):
        return score_derivative(x, xi) - 1

    return brentq(objective, -0.5, 0)

def find_d(xi):
    """
    Encontra d > 0 tal que s'(d) = 0.

    Parameters
    ----------
    xi : float
        Parâmetro de forma, xi > 0.

    Returns
    -------
    d : float
        Ponto de junção da região superior.
    """
    if xi <= 0:
        raise ValueError("find_d requer xi > 0.")

    def objective(x):
        return score_derivative(x, xi)

    # Busca em uma grade de valores positivos
    x_grid = np.linspace(1e-8, 100, 10000)
    values = objective(x_grid)

    # Procura uma mudança de sinal
    for i in range(len(x_grid) - 1):
        if values[i] * values[i + 1] < 0:
            return brentq(
                objective,
                x_grid[i],
                x_grid[i + 1]
            )

    raise ValueError(
        f"Não foi possível encontrar d para xi={xi}."
    )

def composite_score(y, xi, mu=0, varphi=1):
    """
    Composite score do modelo de Harvey.

    Parameters
    ----------
    y : float or np.ndarray
        Observação(ões) y_t.
    mu : float or np.ndarray
        Localização condicional mu_{t|t-1}.
    varphi : float
        Parâmetro de escala, varphi > 0.
    xi : float
        Parâmetro de forma.

    Returns
    -------
    s_dagger : float or np.ndarray
        Composite score s_t^dagger.
    """
    y = np.asarray(y, dtype=float)
    x = _standardized_residual(y, mu, varphi)

    # Caso Gumbel
    if xi == 0:
        return np.where(
            x <= 0,
            x,
            varphi * (1 - np.exp(-x))
        )

    # Caso Fréchet
    if xi > 0:
        c = find_c(xi)
        d = find_d(xi)

        s_composite = np.empty_like(
            np.asarray(x, dtype=float)
        )
        mask_lower = x <= c
        mask_middle = (x > c) & (x <= d)
        mask_upper = x > d
        
        s = score(y[mask_middle], xi, mu, varphi)
        I = fisher_information(xi, varphi)
        s_d = score(mu + varphi * d, xi, mu, varphi)

        s_composite[mask_lower] = x[mask_lower] - c
        s_composite[mask_middle] = s / I
        s_composite[mask_upper] = s_d / I

        return s_composite

    raise ValueError(
        "O composite score não foi definido para xi < 0."
    )

def seasonal_component(t, alpha, beta, period=12):
    """
    Componente sazonal determinística com 6 harmônicos.
    """
    gamma_t = np.zeros_like(np.asarray(t, dtype=float))

    for j in range(1, 7):
        gamma_t += (
            alpha[j - 1] * np.cos(2 * np.pi * j * t / period)
            + beta[j - 1] * np.sin(2 * np.pi * j * t / period)
        )

    return gamma_t

def filter(y, mu0, varphi, phi, kappa, xi, alpha, beta):
    """
    Filtro score-driven do modelo de Harvey com composite score.

    Parameters
    ----------
    y : np.ndarray
        Série temporal de observações y_t.
    mu0 : float
        Nível incondicional da localização.
    varphi : float
        Parâmetro de escala, varphi > 0.
    phi : float
        Parâmetro autorregressivo, |phi| < 1.
    kappa : float
        Parâmetro de resposta do score, kappa > 0.
    xi : float
        Parâmetro de forma da GEV.

    Returns
    -------
    mu_filtered : np.ndarray
        Valores de mu_{t|t-1}.
    x : np.ndarray
        Resíduos padronizados x_t.
    s : np.ndarray
        Composite scores s_t^dagger.
    """
    y = np.asarray(y, dtype=float)

    T = len(y)
    
    mu_filtered = np.empty(T)
    x = np.empty(T)
    s = np.empty(T)
    t = np.arange(1, T + 1)
    gamma_t = seasonal_component(t, alpha, beta)
    
    mu_filtered[0] = mu0
    for t in range(T):

        x[t] = _standardized_residual(y[t], mu_filtered[t], gamma[t], varphi)
        s[t] = composite_score(y[t], xi, mu_filtered[t] + gamma[t], varphi)

        # Atualização do estado
        if t < T - 1:
            mu_filtered[t + 1] = (
                mu0 * (1 - phi)
                + phi * mu_filtered[t]
                + kappa * s[t]
            )

    return mu_filtered, gamma_t, x, s

def simulate(mu0, varphi, phi, kappa, xi, T, burn_in=0, seed=None):
    """
    Simula uma série do modelo de Harvey com composite score.

    Parameters
    ----------
    mu : float
        Nível incondicional da localização.
    varphi : float
        Parâmetro de escala, varphi > 0.
    phi : float
        Parâmetro autorregressivo, |phi| < 1.
    kappa : float
        Parâmetro de resposta do score, kappa > 0.
    xi : float
        Parâmetro de forma da GEV.
    T : int
        Tamanho da série retornada.
    burn_in : int, optional
        Número de observações iniciais descartadas.
    seed : int, optional
        Semente para reprodutibilidade.

    Returns
    -------
    y : np.ndarray
        Série simulada.
    mu_filtered : np.ndarray
        Valores de mu_{t|t-1}.
    x : np.ndarray
        Resíduos padronizados.
    s : np.ndarray
        Composite scores.
    """

    if varphi <= 0:
        raise ValueError("varphi deve ser maior que zero.")

    if abs(phi) >= 1:
        raise ValueError("phi deve satisfazer |phi| < 1.")

    if kappa <= 0:
        raise ValueError("kappa deve ser maior que zero.")

    rng = np.random.default_rng(seed)

    n = T + burn_in
    y = np.empty(n)
    mu_filtered = np.empty(n)
    x = np.empty(n)
    s = np.empty(n)

    mu_filtered[0] = mu0
    epsilon = genextreme.rvs(c=-xi, loc=0, scale=1, size=n, random_state=rng)
    
    for t in range(n):

        y[t] = mu_filtered[t] + varphi * epsilon[t]
        s[t] = composite_score(y[t], xi, mu_filtered[t], varphi)

        if t < n - 1:
            mu_filtered[t + 1] = (
                mu0 * (1 - phi)
                + phi * mu_filtered[t]
                + kappa * s[t]
            )

    # Descartar burn-in
    if burn_in > 0:
        y = y[burn_in:]
        mu_filtered = mu_filtered[burn_in:]
        epsilon = epsilon[burn_in:]
        s = s[burn_in:]

    return y, mu_filtered, epsilon, s

def loglikelihood(y, mu0, varphi, phi, kappa, xi, alpha, beta):
    y = np.asarray(y, dtype=float)

    # Filtragem
    mu_filtered, gamma, x, s = filter(y, mu0, varphi, phi, kappa, xi, alpha, beta)

    # Log-verossimilhança de cada observação
    loglik = logpdf(y, mu_filtered + gamma, varphi, xi)

    # Se alguma observação estiver fora do suporte
    if np.any(~np.isfinite(loglik)):
        return -np.inf

    return np.sum(loglik)

def fit(y, init_params, method="L-BFGS-B"):
    y = np.asarray(y, dtype=float)

    mu0_init, varphi_init, phi_init, kappa_init, xi_init = init_params[:5]
    a_init = np.log(phi_init / (1 - phi_init))
    ratio_init = kappa_init / phi_init
    b_init = np.log(ratio_init / (1 - ratio_init))
    
    def _objective(params):
        mu0, varphi, a_hat, b_hat, xi = params
        phi = expit(a_hat)
        kappa = phi * expit(b_hat)
        
        ll = loglikelihood(y, mu0, varphi, phi, kappa, xi)
        if not np.isfinite(ll):
            return 1e100
        return -ll

    bounds = [
        (None, None),   # mu0
        (1e-8, None),   # varphi > 0
        (None, None),   # a -> phi
        (None, None),   # b -> kappa/phi
        (1e-8, 0.5)     # 0 < xi < 1/2
    ]

    result = minimize(
        _objective,
        x0=np.array([mu0_init, varphi_init, a_init, b_init, xi_init]),
        method=method,
        bounds=bounds
    )

        # recuperar parâmetros no espaço original
    mu0_hat, varphi_hat, a_hat, b_hat, xi_hat = result.x
    phi_hat = expit(a_hat)
    kappa_hat = phi_hat * expit(b_hat)

    result.params_original = np.array([mu0_hat, varphi_hat, phi_hat, kappa_hat, xi_hat])
    return result
    
# def fit(y, init_params, method="L-BFGS-B"):
#     y = np.asarray(y, dtype=float)

#     def _objective(params):
#         mu0, varphi, phi, kappa, xi = params
#         ll = loglikelihood(y, mu0, varphi, phi, kappa, xi)
#         return -ll

#     bounds = [
#         (None, None),  # mu0
#         (1e-8, None),  # varphi > 0
#         (-0.999, 0.999),  # |phi| < 1
#         (1e-8, None),  # kappa > 0
#         (1e-8, 0.999999)   # 0 < xi < 1
#     ]

#     result = minimize(
#         _objective,
#         x0=init_params,
#         method=method,
#         bounds=bounds
#     )
#     return result