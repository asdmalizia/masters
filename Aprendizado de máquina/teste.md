### 1.2 Dinâmica dos parâmetros

Faremos o passo a passo na derivação das dinâmicas do modelo GHP, calcularemos tanto com a função _link_ identidade, como também reparametrizada conforme as funções _link_ abaixo

$$
h_1(\pi_{t|t-1}) = \tilde{\pi}_{t|t-1} = \ln\left( \frac{\pi_{t|t-1}}{1 - \pi_{t|t-1}} \right), \quad
h_2(\lambda_{t|t-1}) = \tilde{\lambda}_{t|t-1} = \ln(\lambda_{t|t-1})
$$

#### 1.2.1 Gradiente da função de probabilidade condicional

Como visto, a função de probabilidade condicional é uma mistura:

$$
\ln(p(y_t \mid \lambda_{t|t-1}, \pi_{t|t-1})) =
\begin{cases}
\ln(\pi_{t|t-1}), & \text{se } y_t = 0 \\\\
\ln(1 - \pi_{t|t-1}) - \lambda_{t|t-1} + y_t \ln(\lambda_{t|t-1}) - \ln(y_t!) - \ln(1 - \exp(-\lambda_{t|t-1})), & \text{se } y_t > 0
\end{cases}
$$

Com isso, temos que

- Para $y_t = 0$:

$$
\nabla_{y_t = 0} =
\begin{bmatrix}
\nabla_{y_t = 0}^{\pi_{t|t-1}} \\
\nabla_{y_t = 0}^{\lambda_{t|t-1}}
\end{bmatrix}
= \begin{bmatrix}
\frac{1}{\pi_{t|t-1}} \\
0
\end{bmatrix}
$$

- Para $y_t > 0$:

$$
\nabla_{y_t > 0} =
\begin{bmatrix}
\nabla_{y_t > 0}^{\pi_{t|t-1}} \\
\nabla_{y_t > 0}^{\lambda_{t|t-1}}
\end{bmatrix}
=\begin{bmatrix}
\frac{-1}{1 - \pi_{t|t-1}} \\\\
\frac{y_t - \lambda_{t|t-1}}{\lambda_{t|t-1}} - \frac{e^{-\lambda_{t|t-1}}}{1 - e^{-\lambda_{t|t-1}}}
\end{bmatrix}
$$

#### 1.2.2 Informação de Fisher

Para a informação de Fisher, temos que

$$
\mathcal{I}_{t|t-1} = \mathbb{E}_{t-1}
\left[
\begin{bmatrix}
(\nabla^{\pi_{t|t-1}})^2 & \nabla^{\pi_{t|t-1}} \nabla^{\lambda_{t|t-1}} \\
\nabla^{\lambda_{t|t-1}} \nabla^{\pi_{t|t-1}} & (\nabla^{\lambda_{t|t-1}})^2
\end{bmatrix}
\right]
=
\begin{bmatrix}
\frac{1}{\pi_{t|t-1}(1 - \pi_{t|t-1})} & 0 \\\\
0 & \frac{ (1 - \pi_{t|t-1}) [ 1 - e^{-\lambda_{t|t-1}} - \lambda_{t|t-1} e^{-\lambda_{t|t-1}} ] }{(1 - e^{-\lambda_{t|t-1}})^2 \lambda_{t|t-1}}
\end{bmatrix}
$$

#### 1.2.3 Jacobiana das funções _link_

Dadas as funções $h_1$ e $h_2$ definidas acima, temos que 

$$
\tilde{h}_{t|t-1} =
\begin{bmatrix}
\frac{\partial h_1}{\partial \pi_{t|t-1}} & 0 \\\\
0 & \frac{\partial h_2}{\partial \lambda_{t|t-1}}
\end{bmatrix} = 
\begin{bmatrix}
\frac{1}{\pi_{t|t-1}(1 - \pi_{t|t-1})} & 0 \\\\
0 & \frac{1}{\lambda_{t|t-1}}
\end{bmatrix}
$$

Com isso, a inversa de $\tilde{h}_{t|t-1}$ é dada por

$$
\tilde{h}_{t|t-1}^{-1} =
\begin{bmatrix}
\pi_{t|t-1}(1 - \pi_{t|t-1}) & 0 \\\\
0 & \lambda_{t|t-1}
\end{bmatrix}
$$

#### 1.2.4 _Score_ do GHP

Encontramos $\nabla_{y_t = 0}$ e $\nabla_{y_t > 0}$, que foram calculadas com base na função _link_ identidade. Agora encontraremos $\tilde{\nabla}_{y_t = 0}$ e $\tilde{\nabla}_{y_t > 0}$. para que possamos encontrar o _score_ e finalmente chegarmos à dinâmica dos parâmetros.

- Para $y_t = 0$:

$$
\tilde{\nabla}_{y_t = 0} =
\tilde{h}_{t|t-1}^{-1} \nabla_{y_t = 0} =
\begin{bmatrix}
1 - \pi_{t|t-1} \\\\
0
\end{bmatrix}
$$

- Para $y_t > 0$:

$$
\tilde{\nabla}_{y_t > 0} =
\tilde{h}_{t|t-1}^{-1} \nabla_{y_t > 0} =
\begin{bmatrix}
- \pi_{t|t-1} \\\\
y_t - \lambda_{t|t-1} - \frac{ \lambda_{t|t-1} e^{-\lambda_{t|t-1}} }{ 1 - e^{-\lambda_{t|t-1}}}
\end{bmatrix}
$$

Abaixo, encontramos a Informação de Fischer reparametrizada:

$$
\tilde{\mathcal{I}}_{t|t-1} =
\tilde{h}_{t|t-1}^{-1} \mathcal{I}_{t|t-1} \tilde{h}_{t|t-1}^{-1}
=
\begin{bmatrix}
\pi_{t|t-1}(1 - \pi_{t|t-1}) & 0 \\\\
0 & \frac{ \lambda_{t|t-1} (1 - \pi_{t|t-1}) [ 1 - e^{-\lambda_{t|t-1}} - \lambda_{t|t-1} e^{-\lambda_{t|t-1}} }{(1 - e^{-\lambda_{t|t-1}})^2}
\end{bmatrix}
$$

Com isso, podemos encontrar os _scores_, conforme equação geral abaixo

$$
\tilde{s}_{t|t-1}
=
\left( \tilde{\mathcal{I}}_{t|t-1} \right)^{-d} \tilde{\nabla}_{t|t-1}
$$

Com isso, temos que

- Para $y_t = 0$:

$$
\tilde{\nabla}_{t|t-1}^{(y=0)} =
\begin{bmatrix}
1 - \pi_{t|t-1} \\
0
\end{bmatrix}
\qquad
\tilde{\mathcal{I}}_{t|t-1} =
\begin{bmatrix}
\pi_{t|t-1}(1 - \pi_{t|t-1}) & 0 \\
0 & \mathcal{I}_{\lambda}
\end{bmatrix}
$$

Logo, o score reparametrizado para d=0 é:

$$
\tilde{s}_{t|t-1}^{(y=0)} =
\begin{bmatrix}
\displaystyle 1 - \pi_{t|t-1} \\
0
\end{bmatrix}
$$

- Para $y_t > 0$:

$$
\tilde{\nabla}_{t|t-1}^{(y>0)} =
\begin{bmatrix}
- \pi_{t|t-1} \\
y_t - \lambda_{t|t-1} - \frac{ \lambda_{t|t-1} e^{-\lambda_{t|t-1}} }{ 1 - e^{-\lambda_{t|t-1}}}
\end{bmatrix}
$$

e

$$
\tilde{\mathcal{I}}_{t|t-1} =
\tilde{h}_{t|t-1}^{-1} \mathcal{I}_{t|t-1} \tilde{h}_{t|t-1}^{-1}
=
\begin{bmatrix}
\pi_{t|t-1}(1 - \pi_{t|t-1}) & 0 \\\\
0 & \frac{ \lambda_{t|t-1} (1 - \pi_{t|t-1}) [ 1 - e^{-\lambda_{t|t-1}} - \lambda_{t|t-1} e^{-\lambda_{t|t-1}} }{(1 - e^{-\lambda_{t|t-1}})^2}
\end{bmatrix}
$$

Para d=0, o score reparametrizado torna-se:

$$
\tilde{s}_{t|t-1}^{(y>0)} =
\begin{bmatrix}
\displaystyle - \pi_{t|t-1} \\
\displaystyle y_t - \lambda_{t|t-1} - \frac{ \lambda_{t|t-1} e^{-\lambda_{t|t-1}} }{ 1 - e^{-\lambda_{t|t-1}}}
\end{bmatrix}
$$

Com isso, utilizamos os scores reparametrizados acima para atualizar $\tilde{\pi}_{t|t-1}$ e $\tilde{\lambda}_{t|t-1}$ via uma estrutura autoregressiva do tipo AR(1) com inovação do tipo GAS.

##### 1.2.4.1 Probabilidade de zero — $\tilde{\pi}_{t|t-1}$

$$
\tilde{\pi}_{t+1|t}
=
\delta_\pi
+
\beta_\pi \cdot \tilde{\pi}_{t|t-1}
+
\rho_\pi \cdot \tilde{s}_{\pi,t|t-1}
$$

onde:

- $\delta_\pi$: constante
- $\beta_\pi$: persistência
- $\rho_\pi$: peso do score
- $\tilde{s}_{\pi,t|t-1}$: componente do score referente a $\pi$

##### 1.2.4.2 Intensidade — $\tilde{\lambda}_{t|t-1}$

$$
\tilde{\lambda}_{t+1|t}
=
\delta_\lambda
+
\beta_\lambda \cdot \tilde{\lambda}_{t|t-1}
+
\rho_\lambda \cdot \tilde{s}_{\lambda,t|t-1}
$$

onde:

- $\tilde{s}_{\lambda,t|t-1}$: componente do score referente a $\lambda$