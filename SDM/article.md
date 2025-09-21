---
title: "Score Driven Models - Lista 1"
subtitle: "Prof. Cristiano Fernandes"
author: "Alessandra Malizia"
date: "Setembro 2025"

lang: pt-BR
fontsize: 12pt
geometry: margin=2cm

toc: true        # gera sumário automático
toc-depth: 2     # profundidade do sumário (até subseções)

lof: false        # lista de figuras
lot: false       # lista de tabelas
---

# Questão 1


$$
\begin{align*}
y_{t|t-1} &= \sigma_{t|t-1} \cdot \epsilon_t \quad \epsilon_t \sim t(0, 1, \nu) \\
\sigma^2_{t+1|t} &= \omega + \alpha y_t^2
\end{align*}
$$



## Item a) Momentos condicionais
![](img/1a.jpg)
   

$$
\frac{(k\omega^2+2k\alpha\omega(\nu\omega/(\nu(1-\alpha)-2)))}{1-k\alpha^2}
$$

## Item b) Distribuição condicional

![](img/1b.jpg)


## Item c) Momentos incondicionais

![](img/1ci.jpg)
![](img/1cii.jpg)
![](img/1ciii.jpg)
![](img/1civ.jpg)
![](img/1cv.jpg)

## Item d) Distribuição incondicional [TODO]

## Item e) Correlações
### i) Linear
![](img/1ei.jpg)

### ii) Quadrática
![](img/1eii.jpg)
![](img/1eiii.jpg)

## Item f) Momentos da distribuição preditiva

![](img/1f.jpg)

## Item g) Distribuição preditiva [TODO]

## Item h) Simulação [TODO]

## Item i) Log verossimilhança

![](img/1i.jpg)


## Item j) Retornos diários [TODO]

# Questão 2
## item a) $  y_t = g_{t|t-1} + \epsilon_t$ 
### i) $ \epsilon_t \sim  Gamma(\mu,\theta)$
![](img/2ai.jpg)

### ii) $ \epsilon_t \sim  Poisson(\lambda)$
![](img/2aii.jpg)

### iii) $ \epsilon_t \sim  Beta(\alpha, \beta)$
![](img/2aiii.jpg)

## item b) $  y_t = g_{t|t-1} \cdot \epsilon_t$ 
### i) $ \epsilon_t \sim  Exp(\lambda)$

![](img/2bi.jpg)

### ii) $ \epsilon_t \sim  Gamma(\mu, \theta)$
![](img/2bii.jpg)

### iii) $ \epsilon_t \sim  Poisson(\lambda)$
![](img/2biii.jpg)