---
layout: post
title: Linear Algebra
categories: ML
---


Linear algebra is a key concept when learning many ML principles and a good understanding will help you a lot.

Most people who are interested in ML have likely studied linear algebra at some point either in high school or university so this post isn't intended as an introduction to linear algebra but more as a refresher. If you haven't learned any linear algebra yet I recommend using [Khan Academy](https://www.khanacademy.org/math/linear-algebra).

$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$
hey
