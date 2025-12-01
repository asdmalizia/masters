---
title: "Score Driven Models - Lista 2"
subtitle: "Prof. Cristiano Fernandes"
author: "Alessandra Malizia"
date: "Novembro de 2025"

lang: pt-BR
fontsize: 12pt
geometry: margin=2cm


lof: false        # lista de figuras
lot: false       # lista de tabelas
pdf-engine: xelatex   
---

# Questão 1


$$
\begin{aligned}
p(y_t|y_{t-1}) \sim NB(\mu, r) \\
\end{aligned}
$$



## Item a) Parametrização e momentos da distribuição

![](img/1a.jpeg)


## Item b) Equação da média com GAS(p,q)
### i) Link identidade
![](img/1bi1.jpg)
![](img/1bi2.jpg)

### ii) Link log
![](img/1bii.jpg)

## Item c) Equação da média com CNO 
$$
\begin{aligned}
p(y_t|y_{t-1}) \sim NB(\mu_{t|t-1}, r) \\
E[y_t|y_{t-1}] = \mu_{t|t-1} \\
h(\mu_{t+1|t}) = \tilde\mu_{t+1|t} \\
\tilde\mu_{t+1|t} = m_{t+1|t} + \gamma_{t+1|t}
\end{aligned}
$$

### i) Sazonalidade por dummy Harrison & Stevens
![](img/1ci.jpeg)

### ii) Sazonalidade por trigonométricos
A tendência $m_{t+1|t}$ segue um AR(1) com drift, enquanto a sazonalidade é dada pela soma de funções trigonométricas. É possível escrever as equações dessas componentes não observáveis na forma matricial abaixo.

![](img/formula1c.jpg)

Como calculado anteriormente, a função score na forma matricial acima pode ser escrita segundo as equações abaixo.
$$
\begin{aligned}
\tilde s_{t} = y_te^{-\tilde \mu_{t|t-1}}-1 \\
\Rightarrow \tilde s_{t} = y_te^{-m_{t|t-1} + \gamma_{t|t-1}}-1
\end{aligned}
$$

## Item d) Log da verossimilhança
O log da verossimilhança foi calculado anteriormente em relação aos parâmetros variáveis do modelo para o cálculo da função score. Usando os mesmos resultados e observando a função da log verossimilhança para os parâmetros fixos do modelo e suas restrições, é possível ober os resultados abaixo.

![](img/1d.jpeg)


# Questão 2
## Item a)
![](img/2a.jpeg)

## Item b)
![](img/2b.jpeg)

# Questão 3
$$
\begin{aligned}
y_{t} &= \mu_t + \epsilon_t,  \quad \epsilon_t \sim N(0, \sigma^2) \\
\end{aligned}
$$

![](img/3.jpg)

# Questão 4

## Item a) Parametrização e momentos da distribuição
![](img/4ai.jpeg)

![](img/4aii.jpeg)

## Item b) Equação dos parâmetros
### i) GAS 
![](img/4bia.jpeg)

![](img/4bib.jpeg)

![](img/4bic.jpeg)

### ii) CNO

O parâmetro da média condicional é descrito por uma componente de tendência AR(1) e uma componente de sazonalidade por trigonométricos, confofrme equações abaixo. A função score foi calculada anteriormente. 

![](img/4bii.jpeg)

## Item c) Log da verossimilhança
![](img/4c.jpeg)

## Item d) Estimação do modelo

Paa estimar o modelo, foi usada uma série diária de armazenamento de energia na região sudeste. A série é disponibilizada em megawatts pelo Operador Nacional de Sistema Elétrico (ONS), conforme os metadados abaixo. Para obter uma série mensal, foi feita a média mensal dos dados diários.
![](img/metadados.png)

Como pode ser observado nos gráficos, a série apresenta suporte positivo e forte sazonalidade. O modelo escolhido foi a densidade condicional de Weibull com média variante no tempo e parâmetro fixo de forma.

![](img/dados.png)

A média condicional foi modelada por uma componente de tendência linear e sazonalidade por dummies. As componentes estimadas podem ser vistas nos resultados abaixo.

![](img/componentes.png)
![](img/estimação.png)

Conforme observado nos gráficos acima e nos valores estimados, o parâmetro de atualização da sazonalidade é próximo de zero, indicando que a sazonalidade possui comportamento praticamente fixo ao longo dos anos. Já o parâmetrro de forma da distribuição de Weibull é relativamente alto, indicando uma baixa assimetria nos dados. 

Nos resultados do diagnóstico abaixo, é possível observar que o modelo captou bem a dinâmica dos dados, uma vez que os resíduos padronizados apresentam baixa autocorrelação nos lags. O qq-plot e o histograma indicam uma distribuição próxima da normalidade, mas apresentam alumas observações aberrantes nas caudas. Esse resultado pode indicar a necessidade de tratamento de outliers.

![](img/diagnóstico.png)

O modelo estimado foi comparado com um modelo auto ARIMA na amostra de teste. O ARIMA escolhido foi o ARMA(12,1,1). O modelo score driven apresentou melhor desempenho fora da amostra em termos de RMSE e MAE, conforme tabela abaixo.

![](img/forercasts.png)