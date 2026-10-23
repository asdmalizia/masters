import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import gamma as gamma_func, expit
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

gamma_E = 0.5772156649015329

def _standardized_residual(y, mu, gamma, varphi):
    """
    Calcula o resíduo padronizado:

        x_t = (y_t - mu_{t|t-1}) / varphi

    Returns
    -------
    x : float or np.ndarray
        Resíduo padronizado x_t.
    """
    return (y - mu - gamma) / varphi

def _xi_t_from_dummies(xi_seca, xi_cheia, dummy):
    """
    Constrói o parâmetro de forma condicional xi_t
    a partir das dummies de seca e cheia.

    xi_t = xi_seca + D_cheia * xi_cheia
    """
    dummy = np.asarray(dummy, dtype=float)
    return xi_seca + dummy * (xi_cheia - xi_seca)

def _varphi_t_from_dummies(varphi_seca, varphi_cheia, dummy):
    dummy = np.asarray(dummy, dtype=float)
    return varphi_seca + dummy * (varphi_cheia - varphi_seca)

def pdf(y, mu, varphi, xi, gamma=0):
    """
    Densidade GEV condicional.

    Returns
    -------
    p : float or np.ndarray
        Densidade condicional.
    """
    return np.exp(logpdf(y, mu, varphi, xi, gamma))

def cdf(y, mu, varphi, xi, gamma=0):
    """
    CDF condicional da GEV com sazonalidade.
    F_t(y) = P(Y_t <= y | F_{t-1})

    Retorna
    -------
    float ou np.ndarray
        Probabilidade acumulada condicional.
    """

    y, mu, varphi, xi, gamma = np.broadcast_arrays(
        np.asarray(y, dtype=float),
        np.asarray(mu, dtype=float),
        np.asarray(varphi, dtype=float),
        np.asarray(xi, dtype=float),
        np.asarray(gamma, dtype=float)
    )
    if np.any(~np.isfinite(varphi)) or np.any(varphi <= 0):
        raise ValueError("varphi deve ser positivo e finito.")

    # Resíduo padronizado
    x = _standardized_residual(y, mu, gamma, varphi)
    z = 1 + xi * x

    F = np.empty_like(x, dtype=float)

    # Caso Gumbel: xi_t = 0
    mask0 = np.abs(xi) < 1e-6
    F[mask0] = np.exp(-np.exp(-x[mask0]))

    # Caso xi_t diferente de zero
    mask1 = ~mask0
    valid = mask1 & (z > 0)

    F[valid] = np.exp(
        -np.exp(-np.log(z[valid]) / xi[valid])
    )
    # Fora do suporte da GEV
    F[mask1 & (z <= 0) & (xi > 0)] = 0.0
    F[ mask1 & (z <= 0) & (xi < 0)] = 1.0
    return F

def ppf(p, mu, varphi, xi, gamma=0):
    """
    Função quantil (PPF) condicional da GEV com sazonalidade.
    Q_t(p) = F_t^{-1}(p)

    Retorna
    -------
    float ou np.ndarray
        Quantil condicional na escala original de y_t.
    """

    p, mu, varphi, xi, gamma = np.broadcast_arrays(
        np.asarray(p, dtype=float),
        np.asarray(mu, dtype=float),
        np.asarray(varphi, dtype=float),
        np.asarray(xi, dtype=float),
        np.asarray(gamma, dtype=float)
    )

    if np.any(~np.isfinite(varphi)) or np.any(varphi <= 0):
        raise ValueError("varphi deve ser positivo e finito.")

    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("p deve estar estritamente entre 0 e 1.")

    q = np.empty_like(p, dtype=float)

    # Caso Gumbel: xi_t aproximadamente zero
    mask0 = np.abs(xi) < 1e-6
    q[mask0] = -np.log(-np.log(p[mask0]))

    # Caso xi_t diferente de zero
    mask1 = ~mask0
    q[mask1] = (
        np.expm1(-xi[mask1] * np.log(-np.log(p[mask1])))
        / xi[mask1]
    )

    result = mu + gamma + varphi * q
    return result

def conditional_mean(mu, varphi, xi, gamma=0):
    """
    Média condicional da GEV.

    E[y_t | F_{t-1}].
    """

    mu, varphi, xi, gamma = np.broadcast_arrays(
        np.asarray(mu, dtype=float),
        np.asarray(varphi, dtype=float),
        np.asarray(xi, dtype=float),
        np.asarray(gamma, dtype=float)
    )
    mean = np.empty_like(mu, dtype=float)

    # Gumbel
    mask0 = np.abs(xi) < 1e-6
    mean[mask0] = mu[mask0] + gamma[mask0] + varphi[mask0] * gamma_E

    # Frechet
    mask1 = ~mask0
    if np.any(xi[mask1] >= 1):
        raise ValueError("A média condicional não existe para xi >= 1.")
        
    mean[mask1] = (
        mu[mask1]
        + gamma[mask1]
        + varphi[mask1]
        * (
            gamma_func(1 - xi[mask1]) - 1
        )
        / xi[mask1]
    )
    return mean

def conditional_variance(varphi, xi):
    """
    Variância condicional da GEV.

    Var(y_t | F_{t-1}).
    """

    varphi, xi = np.broadcast_arrays(
        np.asarray(varphi, dtype=float),
        np.asarray(xi, dtype=float)
    )
    variance = np.empty_like(varphi, dtype=float)

    # Gumbel
    mask0 = np.abs(xi) < 1e-6
    variance[mask0] = np.pi**2 / 6 * varphi[mask0]**2
    
    # Frechet
    mask1 = ~mask0
    if np.any(xi[mask1] >= 0.5):
        raise ValueError("A variância condicional não existe para xi >= 1/2.")

    variance[mask1] = (
        varphi[mask1]**2
        / xi[mask1]**2
        * (
            gamma_func(1 - 2 * xi[mask1])
            - gamma_func(1 - xi[mask1])**2
        )
    )
    return variance

def logpdf(y, mu, varphi, xi, gamma=0):
    """
    Log-densidade GEV condicional.

    Returns
    -------
    log_p : float or np.ndarray
        Log-densidade condicional.
    """
    y, mu, varphi, xi, gamma = np.broadcast_arrays(
        np.asarray(y, dtype=float),
        np.asarray(mu, dtype=float),
        np.asarray(varphi, dtype=float),
        np.asarray(xi, dtype=float),
        np.asarray(gamma, dtype=float)
    )

    x = _standardized_residual(y, mu, gamma, varphi)
    log_p = np.full_like(x, -np.inf, dtype=float)
    
    # caso Gumbel
    mask0 = np.abs(xi) < 1e-6
    safe = mask0 & (x > -700)
    log_p[safe] = (
        -np.log(varphi[safe]) - x[safe] - np.exp(-x[safe])
    )
    
    # caso Frechet
    mask1 = ~mask0
    z = 1 + xi * x
    valid = mask1 & (z > 0)
    
    log_p[valid] = (
        -np.log(varphi[valid])
        - (
            1 + 1 / xi[valid]
          ) * np.log(z[valid])
        - np.exp(
            -np.log(z[valid]) / xi[valid]
        )
    )
    return log_p

        

def score(y, xi, mu=0, gamma=0, varphi=1):
    """
    Score da log-densidade em relação a mu_{t|t-1}.

    Returns
    -------
    score : float or np.ndarray
        Score em relação a mu_{t|t-1}.
    """
    # y = np.asarray(y, dtype=float)
    # xi = np.asarray(xi, dtype=float)
    y, xi, varphi, mu, gamma = np.broadcast_arrays(
        np.asarray(y, dtype=float),
        np.asarray(xi, dtype=float),
        np.asarray(varphi, dtype=float),
        np.asarray(mu, dtype=float),
        np.asarray(gamma, dtype=float)
    )
    x = _standardized_residual(y, mu, gamma, varphi)
    
    result = np.full_like(x, np.nan)
    
    # caso Gumbel
    mask0 = np.abs(xi) < 1e-6
    result[mask0] = (
        1 / varphi[mask0]
        * (1 - np.exp(-x[mask0]))
    )
    
    # caso Frechet
    mask1 = ~mask0
    z = 1 + xi * x
    
    valid = mask1 & (z > 0)
    result[valid] = (
        1 / varphi[valid]
        * (
            (1 + xi[valid]) / z[valid]
            - np.exp(
                -(1 / xi[valid] + 1) * np.log(z[valid])
            )
        )
    )
    return result



def fisher_information(xi, varphi=1):
    """
    Informação de Fisher em relação a mu_{t|t-1}.

    Returns
    -------
    I : float
        Informação de Fisher para mu_{t|t-1}.
    """
    
    xi, varphi = np.broadcast_arrays(
        np.asarray(xi, dtype=float),
        np.asarray(varphi, dtype=float)
    )
    I = np.empty_like(xi, dtype=float)

    mask0 = np.abs(xi) < 1e-6
    I[mask0] = 1 / varphi[mask0] ** 2
    
    mask1 = ~mask0
    I[mask1] = (
            (1 + xi[mask1]) ** 2
            * gamma_func(1 + 2 * xi[mask1])
            / varphi[mask1] ** 2
        )

    return I

def score_derivative(x, xi):
    """
    Derivada do score padronizado em relação a x (resíduo padronizado).

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
            (1 + xi) * gamma_func(1 + 2 * xi)
        )
        * (
            xi / z**2 - 1 / z**(2 + 1 / xi)
        )
    )


def _find_c(xi):
    """
    Encontra c tal que s'(c) = 1.
    Para 0 < xi < 1, a raiz está em aproximadamente
    [-0.5, 0].
    """
    # if not 0 < xi < 1:
    #     raise ValueError("find_c requer 0 < xi < 1.")

    def objective(x):
        return score_derivative(x, xi) - 1
    # lower, upper = -1 / xi + 1e-10, 1e-10
    return brentq(objective, -0.5, 0)

# def _find_d(xi):
#     """
#     Encontra d > 0 tal que s'(d) = 0.

#     Parameters
#     ----------
#     xi : float
#         Parâmetro de forma, xi > 0.

#     Returns
#     -------
#     d : float
#         Ponto de junção da região superior.
#     """
#     if xi <= 0:
#         raise ValueError("find_d requer xi > 0.")

#     def objective(x):
#         return score_derivative(x, xi)

#     # Busca em uma grade de valores positivos
#     x_grid = np.linspace(1e-8, 100, 10000)
#     values = objective(x_grid)

#     # Procura uma mudança de sinal
#     for i in range(len(x_grid) - 1):
#         if values[i] * values[i + 1] < 0:
#             return brentq(
#                 objective,
#                 x_grid[i],
#                 x_grid[i + 1]
#             )

#     raise ValueError(
#         f"Não foi possível encontrar d para xi={xi}."
#     )

def _find_d(xi):
    """
    Calcula d, ponto em que a derivada do score é zero.

    Para xi > 0:
        d = (xi^(-xi) - 1) / xi
    """

    if xi <= 0:
        raise ValueError("xi deve ser positivo.")

    return np.expm1(-xi * np.log(xi)) / xi

def composite_score(y, xi, mu=0, gamma=0, varphi=1, c={}, d={}, I={}):
    """
    Composite score do modelo de Harvey.

    Returns
    -------
    s_dagger : float or np.ndarray
        Composite score s_t^dagger.
    """
    # y = np.asarray(y, dtype=float)
    # xi = np.asarray(xi, dtype=float)
    y, xi, mu, varphi, gamma = (
        a.ravel() for a in (y, xi, mu, varphi, gamma)
    )
    x = _standardized_residual(y, mu, gamma, varphi)

    # s_composite = np.empty_like(x, dtype=float)
    s_composite = np.full_like(x, np.nan, dtype=float)

    for i in range(len(y)):
        xi_i = xi[i]
        varphi_i = varphi[i]
        x_i = x[i]

        if not np.isfinite(varphi_i) or varphi_i <= 0:
            continue

        # Caso Gumbel: preserva a regra da implementação original
        if abs(xi_i) < 1e-6:
            s_composite[i] = varphi_i * x_i
            if x_i > 0:
                s_composite[i] = varphi_i * (1 - np.exp(-x_i))
            continue

        z_i = 1 + xi_i * x_i
        if z_i <= 0:
            continue

        if xi_i not in c:
            c[xi_i] = _find_c(xi_i)
        if xi_i not in d:
            d[xi_i] = _find_d(xi_i)

        key_I = (float(xi_i), float(varphi_i))
        if key_I not in I:
            I[key_I] = float(fisher_information(xi_i, varphi_i))

        c_i = c[xi_i]
        d_i = d[xi_i]
        I_i = I[key_I]

        if x_i <= c_i:
            s_composite[i] = varphi_i * (x_i - c_i)

        elif x_i <= d_i:
            s_i = score(
                y[i], xi_i, mu[i], gamma[i], varphi_i
            )
            s_composite[i] = float(s_i) / I_i

        else:
            s_d = score(
                mu[i] + gamma[i] + varphi_i * d_i,
                xi_i, mu[i], gamma[i], varphi_i
            )
            s_composite[i] = float(s_d) / I_i

    return s_composite


def seasonal_component(t, alpha, beta, period=12):
    """
    Componente sazonal determinística com 6 harmônicos.
    """
    gamma_t = np.zeros_like(np.asarray(t, dtype=float))

    for j in range(1, 6):
        gamma_t += (
            alpha[j - 1] * np.cos(2 * np.pi * j * t / period)
            + beta[j - 1] * np.sin(2 * np.pi * j * t / period)
        )  
    # 6º harmônico: somente cosseno
    gamma_t += alpha[5] * np.cos(2 * np.pi * 6 * t / period)

    return gamma_t

def initial_seasonal_coefficients(y, period=12, n_harmonics=6):

    y = np.asarray(y, dtype=float)
    t = np.arange(1, len(y) + 1)
    X = [np.ones(len(y))]

    for j in range(1, n_harmonics):
        X.append(np.cos(2 * np.pi * j * t / period))
        X.append(np.sin(2 * np.pi * j * t / period))
    X.append(np.cos(2 * np.pi * n_harmonics * t / period))
    X = np.column_stack(X)

    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    mu_init = coefficients[0]
    alpha = np.asarray(coefficients[1::2]).ravel()
    beta = np.asarray(coefficients[2::2]).ravel()

    return mu_init, alpha, beta

def filter(y, mu0, varphi_seca, varphi_cheia, phi, kappa, 
    xi_seca, xi_cheia, dummy, alpha, beta):
    """
    Filtro score-driven do modelo de Harvey com composite score e sazonalidade.

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

    time = np.arange(1, T + 1)
    gamma_t = seasonal_component(time, alpha, beta)
    varphi_t = _varphi_t_from_dummies(varphi_seca, varphi_cheia, dummy)
    xi_t = _xi_t_from_dummies(xi_seca, xi_cheia, dummy)

    c, d = {}, {}
    I = {}
    for (xi_value, varphi_value) in [(xi_seca, varphi_seca), (xi_cheia, varphi_cheia)]:
        if xi_value > 1e-6:
            c[xi_value] = _find_c(xi_value)
            d[xi_value] = _find_d(xi_value)
        I[xi_value] = fisher_information(xi_value, varphi_value)
    
        
    mu_filtered = np.empty(T)
    x = np.empty(T)
    s = np.empty(T)
    
    mu_filtered[0] = mu0
    for t in range(T):

        x[t] = _standardized_residual(y[t], mu_filtered[t], gamma_t[t], varphi_t[t])
        s[t] = composite_score(y[t], xi_t[t], mu_filtered[t], gamma_t[t], varphi_t[t], c, d, I)

        # Atualização do estado
        if t < T - 1:
            mu_filtered[t + 1] = (
                mu0 * (1 - phi)
                + phi * mu_filtered[t]
                + kappa * s[t]
            )

    return mu_filtered, gamma_t, x, s, xi_t, varphi_t

def simulate(mu0, varphi_seca, varphi_cheia, phi, kappa, xi_seca, xi_cheia, 
             dummy, alpha, beta, T, burn_in=0, seed=None):
    """
    Simula uma série do modelo de Harvey com composite score.

    Parameters
    ----------
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

    if varphi_seca <= 0 or varphi_cheia <= 0:
        raise ValueError("varphi deve ser maior que zero.")

    if abs(phi) >= 1:
        raise ValueError("phi deve satisfazer |phi| < 1.")

    if kappa <= 0:
        raise ValueError("kappa deve ser maior que zero.")

    rng = np.random.default_rng(seed)
    dummy = np.asarray(dummy, dtype=float)

    n = T + burn_in
    y = np.empty(n)
    mu_filtered = np.empty(n)
    epsilon = np.empty(n)
    s = np.empty(n)
 
    time = np.arange(1, n + 1)
    gamma = seasonal_component(time, alpha, beta)
    varphi_t = _varphi_t_from_dummies(varphi_seca, varphi_cheia, dummy)
    xi_t = _xi_t_from_dummies(xi_seca, xi_cheia, dummy)

    # Valores de c, d e I para cada xi
    c, d, I = {}, {}, {}
    for (xi_value, varphi_value) in [(xi_seca, varphi_seca), (xi_cheia, varphi_cheia)]:
        c[xi_value] = None
        d[xi_value] = None
        
        if xi_value > 1e-6:
            c[xi_value] = _find_c(xi_value)
            d[xi_value] = _find_d(xi_value)
        I[xi_value] = fisher_information(xi_value, varphi_value)

    # filtro
    mu_filtered[0] = mu0    
    for t in range(n):

        epsilon[t] = genextreme.rvs(c=-xi_t[t], loc=0, scale=1, random_state=rng)
        y[t] = mu_filtered[t] + gamma[t] + varphi_t[t] * epsilon[t]
        s[t] = composite_score(y[t], xi_t[t], mu_filtered[t], gamma[t], varphi_t[t], c, d, I)

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
        gamma = gamma[burn_in:]
        epsilon = epsilon[burn_in:]
        s = s[burn_in:]
        xi_t = xi_t[burn_in:]
        varphi_t = varphi_t[burn_in:]

    return y, mu_filtered, gamma, epsilon, s, xi_t, varphi_t

def loglikelihood(y, mu0, varphi_seca, varphi_cheia, 
                  phi, kappa, xi_seca, xi_cheia, dummy, alpha, beta):
    y = np.asarray(y, dtype=float)

    # Filtragem
    mu_filtered, gamma, x, s, xi_t, varphi_t = filter(y, mu0, varphi_seca, varphi_cheia, phi, kappa, xi_seca, xi_cheia, dummy, alpha, beta)

    # Log-verossimilhança de cada observação
    loglik = logpdf(y, mu_filtered, varphi_t, xi_t, gamma)

    # Se alguma observação estiver fora do suporte
    if np.any(~np.isfinite(loglik)):
        return -np.inf

    return np.sum(loglik)

def _unpack(params, xi_mode, varphi_mode):
    # Parâmetros sempre na ordem em que são otimizados
    pos = 0
    mu0 = params[pos]; pos += 1
    varphi_seca = params[pos]; pos += 1
    phi = params[pos]; pos += 1
    kappa = params[pos]; pos += 1
    xi_seca = params[pos]; pos += 1

    if varphi_mode == "seasonal":
        varphi_cheia = params[pos]
        pos += 1
    else:
        varphi_cheia = varphi_seca

    if xi_mode == "seasonal":
        xi_cheia = params[pos]
        pos += 1
    else:
        xi_cheia = xi_seca

    alpha = params[pos:pos + 6]
    pos += 6
    beta = params[pos:pos + 5]

    return (
        mu0, varphi_seca, varphi_cheia, phi, kappa,
        xi_seca, xi_cheia, alpha, beta
    )

def fit(y, dummy, init_params, method="L-BFGS-B", bounds=None, 
        xi_mode="seasonal", varphi_mode="seasonal",):
    
    y = np.asarray(y, dtype=float)
    
    # Ordem fixa do vetor inicial:
    # mu0, varphi_seca, varphi_cheia, phi, kappa,
    # xi_seca, xi_cheia, alpha[6], beta[5]
    (
        mu0_init, varphi_seca_init, varphi_cheia_init,
        phi_init, kappa_init, xi_seca_init, xi_cheia_init
    ) = init_params[:7]

    alpha_init = init_params[7:13]
    beta_init = init_params[13:18]

    x0 = [mu0_init, varphi_seca_init, phi_init, kappa_init, xi_seca_init]
    default_bounds = [
        (None, None),  # mu0
        (1e-8, None),  # varphi > 0
        (-0.999, 0.999),  # -1 < phi < 1
        (1e-8, None),  # kappa > 0
        (1e-8, 0.5)   # xi_seca
    ]  
 
    if varphi_mode == "seasonal":
        x0.append(varphi_cheia_init)
        default_bounds.append((1e-8, None))  # varphi_cheia > 0 
    if xi_mode == "seasonal":
        x0.append(xi_cheia_init)
        default_bounds.append((1e-8, 0.5))    # xi_cheia
                
    x0 += [*alpha_init, *beta_init]    
    x0 = np.array(x0, dtype=float)    
    default_bounds.extend([(None, None)] * 11)  # alpha e beta   
    if bounds is None:
        bounds = default_bounds
        
    def _objective(params):
        (
            mu0, varphi_seca, varphi_cheia, phi, kappa,
            xi_seca, xi_cheia, alpha, beta
        ) = _unpack(params, xi_mode, varphi_mode)

        ll = loglikelihood(
            y, mu0, varphi_seca, varphi_cheia, phi, kappa,
            xi_seca, xi_cheia, dummy, alpha, beta
        )
        if not np.isfinite(ll):
            return 1e100
        return -ll

    result = minimize(_objective, x0=x0, method=method, bounds=bounds)

    (
        mu0_hat, varphi_seca_hat, varphi_cheia_hat, phi_hat, kappa_hat,
        xi_seca_hat, xi_cheia_hat, alpha_hat, beta_hat
    ) = _unpack(result.x, xi_mode, varphi_mode)


    result.params_original = np.array([
        mu0_hat, varphi_seca_hat, varphi_cheia_hat, 
        phi_hat, kappa_hat,
        xi_seca_hat, xi_cheia_hat,
        *alpha_hat, *beta_hat
    ])
    result.param_names = [
        "mu0", "varphi_seca", "varphi_cheia", "phi", "kappa",
        "xi_seca", "xi_cheia",
        "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6",
        "beta1", "beta2", "beta3", "beta4", "beta5"
    ]

    result.xi_mode = xi_mode
    result.varphi_mode = varphi_mode
    return result
    