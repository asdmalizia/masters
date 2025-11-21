---
title: "Score Driven Models - Lista 2"
subtitle: "Prof. Cristiano Fernandes"
author: "Alessandra Malizia"
date: "Outubro de 2025"

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

$$
\begin{aligned}
\tilde s_{t} = y_te^{-\tilde \mu_{t|t-1}}-1 \\
\Rightarrow \tilde s_{t} = y_te^{-m_{t|t-1} + \gamma_{t|t-1}}-1
\end{aligned}
$$

## Item d) Log da verossimilhança
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
![](img/4bii.jpeg)

## Item c) Log da verossimilhança
![](img/4c.jpeg)

## Item d) Estimação do modelo [TODO]