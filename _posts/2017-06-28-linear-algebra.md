---
layout: post
title: Linear Algebra
categories: ML
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
Linear algebra is a key concept when learning many ML principles and a good understanding will help you a lot.

Most people who are interested in ML have likely studied linear algebra at some point either in high school or university so this post isn't intended as an introduction to linear algebra but more as a refresher. If you haven't learned any linear algebra yet I recommend using [Khan Academy](https://www.khanacademy.org/math/linear-algebra).

Linear Algebra specifically entails the use of matrices and vectors for...

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

this is the text in between equations and code

{% highlight matlab %}
function [centres, data] = meanshift(X, lambda)
%MEANSHIFT mean shift implementation function using a flat kernel
%   
%   For COMP4702 - Prac 4 Question 2
%   By James Copperthwaite - 44312042
%
% [centres, data] = meanshift(X, lambda) takes these arguments:
%
%   X = n x d matrix of doubles
%   lambda = double, non-negative radius paramater
%
% and returns:
%   centres = q x d matrix
%   data = 1 x q cell containing the clustered points for each centre
%
% d refers to the dimensionality of the dataset
% n is the number of the datapoints in the dataset
% q is the number of centres

    [n, d] = size(X);    % obtaining dataset dimensions and length


    K = @(x) (norm(x)<=lambda); % Flat kernel

    % initial guess slects random element
    %m_i = X(ceil(rand * n), :)

    centre_m = [];
    class = {};

    for j = 1:n
        m_i = X(j,:);

        mt = 0;
        mb = 0;
        m_p = m_i/2;
        while m_p~=m_i
            for i = 1:n
                mt = mt + (X(i,:) * K(m_i - X(i,:)));
                mb = mb + K(X(i,:) - m_i);
            end
            m_p = m_i;
            m_i = mt/mb;

            mt = 0;
            mb = 0;
        end

        %find similar value
        if ~isempty(centre_m)
            [temp, indx] = min(abs(bsxfun(@minus,centre_m,m_i)));

            if norm(temp)> 0.5
                centre_m = [centre_m; m_i];
                class{size(class,2)+1} = [X(j,:)];
            else
                t=indx(1);
                class{t} = [class{t}; X(j,:)];
            end
        else
            centre_m = [centre_m; m_i];
            class{size(class,2)+1} = [X(j,:)];
        end

    end
    centres = centre_m;
    data = class;
end
{% endhighlight %}

that was matlab code
