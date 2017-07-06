---
layout: post
title: Linear Algebra
categories: ML
---
<style>
#ppimg {
    display: block;
    margin-left: auto;
    margin-right: auto }
</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
Linear algebra is a key concept when learning many ML principles and a good understanding will help you a lot.

Most people who are interested in ML have likely studied linear algebra at some point either in high school or university so this post isn't intended as an introduction to linear algebra but more as a refresher. If you haven't learned any linear algebra yet I recommend using [Khan Academy](https://www.khanacademy.org/math/linear-algebra).

Linear Algebra typically entails the use of matrices and vectors to represent space and data. In machine learning it is used for things like data manipulation, image recognition, dimensionality reduction and many others.

In this post I'll cover the following main topics relevant to machine learning:
1. Vectors and Matrices
2. Matrix Operations
3. The Unit Vector
4. Matrix Identities, Inverses and Determinants
5. Dot and Cross Products
6. Transformations
7. Eigenvectors/Eigenvalues


## 1. Vectors and Matrices

Vectors and matrices form the foundations of linear algebra and are incredibly useful for many things not just machine learning.

### Vectors

A vector is a mathematical object that can be represented in multiple ways; graphically, as a tuple/set and as cartesian components. They are defined as having a magnitude and direction, and as such vectors with the same magnitude and direction are considered equal **even if they do not have the same basis** in other words parallel vectors of the same magnitude are equal. Vectors themselves exist in *space* whether this be euclidian space, vector space or any other form of mathematical space, euclidean space is the most commonly used and most useful in engineering and machine learning.

<table>
  <thead>
    <tr>
      <th>Vector Notation</th>
      <th>Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Graphical</td>
      <td>
      <figure>
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/95/Vector_from_A_to_B.svg" width="250" alt="vector graphic of a vector" id="ppimg"/>
        <figcaption>(image from wikipedia)</figcaption>
      </figure>
      </td>
    </tr>
    <tr>
      <td>Tuple/Set (vertical)</td>
      <td>$$ \vec{v} =  \begin{pmatrix}
                          7 \\
                          4 \\
                          5 \\
                          \end{pmatrix} $$</td>
    </tr>
    <tr>
      <td>Tuple/Set (horizontal)</td>
      <td>$$ \vec{v} = (7, 4, 5) $$</td>
    </tr>
    <tr>
      <td>Cartesian components</td>
      <td>$$ \vec{v} = 7 \mathbf{\widehat{i}} + 4 \mathbf{\widehat{i}} + 5 \mathbf{\widehat{k}} $$</td>
    </tr>
  </tbody>
</table>

Vectors themselves can be written in a few ways,  as a bold lower-case letter, a lower-case letter with a little arrow hat or as two capitals with an arrow hat denoting a vector from one point to another (e.g. point O to point A). Sometimes vectors can be written as a lowercase with a squiggle/tilde under them however this is usually handwritten shorthand.

$$\mathbf{v} = \vec{v} = \overrightarrow{OA}$$

In terms of specific vector representations used in Machine learning, most commonly you see horizontal and vertical tuples used as they are the easiest to work with when programming. In terms of my experience reading textbooks the vectors themselves are usually denoted by a meaning full lower-case bold letter (such as $$\mathbf{w}$$ for weights) and are either horizontal or vertical depending on how they are used (i.e. so that the matrix dimensions agree).

### Matrices

A Matrix  is a collection or **array** of numbers with rows and columns, matrices can be used to represent many things, space, data-points, images and others.

In machine learning matrices are used to speed process up, e.g. many GPU's used for training data have specific instructions to perform Matrix multiplications. For example in a Perceptron model, rather than  individually applying the weights to the inputs, it uses matrices instead. Other obvious reasons to use matrices is for cleaner code as they are a nice way to store data.

Matrices are denoted by a upper-case  letter (unlike a vector) and may be seen in papers/textbooks with square, round or no brackets.

$$ \mathbf{A} =  \begin{matrix}
                    1 & 2 & 3 \\
                    4 & 5 & 6 \\
                    7 & 8 & 9 \\
                    \end{matrix} = \begin{pmatrix}
                                        1 & 2 & 3 \\
                                        4 & 5 & 6 \\
                                        7 & 8 & 9 \\
                                        \end{pmatrix} = \begin{bmatrix}
                                                            1 & 2 & 3 \\
                                                            4 & 5 & 6 \\
                                                            7 & 8 & 9 \\
                                                            \end{bmatrix} $$

The numbers inside an array are usually referred to as elements (like arrays in programming) and are usually use lowercase letters to denote the index of these elements. It is also important to remeber that when referring to an array it is referenced as rows by columns ( $$ m\times n$$ array of m rows and n columns). This is sometimes a source of error as it can be interpreted similar to cartesian ( $$ x\times y$$).

Another note is that indexing starts at 1 and not 0 like in arrays, matlab uses indexing at 1 however most other languages index at 0 so it's an important thing to consider when translating equations from papers to actual code.

$$
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} \\
\end{pmatrix}
$$

## 2. Matrix Operations

Matrices can be used in equations (hence linear algebra...) however cannot always be treated in the same ways you would a pro-numeral in algebra. Below is a table of example matrix Operations

<table>
  <thead>
    <tr>
      <th>Matrix Operation</th>
      <th>Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Scalar Addition</td>
      <td>$$ \begin{bmatrix}
              1 & 2 \\
              3 & 4 \\
              \end{bmatrix} + 1 = \begin{bmatrix}
                                  2 & 3 \\
                                  4 & 5 \\
                                  \end{bmatrix}
          $$
      </td>
    </tr>
    <tr>
      <td>Scalar Multiplication </td>
      <td>$$ \begin{bmatrix}
              1 & 2 \\
              3 & 4 \\
              \end{bmatrix} \times 2 = \begin{bmatrix}
                                  2 & 4 \\
                                  6 & 8 \\
                                  \end{bmatrix}
          $$
      </td>
    </tr>
    <tr>
      <td>Matrix Addition<sup>1</sup></td>
      <td>$$ \begin{bmatrix}
              1 & 2 \\
              3 & 4 \\
              \end{bmatrix} + \begin{bmatrix}
                                  2 & 4 \\
                                  6 & 8 \\
                                  \end{bmatrix} = \begin{bmatrix}
                                                      3 & 6 \\
                                                      9 & 12 \\
                                                      \end{bmatrix}
          $$
      </td>
    </tr>
    <tr>
      <td>Matrix Multiplication<sup>2</sup></td>
      <td>$$ \begin{bmatrix}
              1 & 2 \\
              3 & 4 \\
              \end{bmatrix} \times \begin{bmatrix}
                                  2 \\
                                  3 \\
                                  \end{bmatrix} = \begin{bmatrix}
                                                      8 \\
                                                      18 \\
                                                      \end{bmatrix}
          $$
      </td>
    </tr>
  </tbody>
</table>
<p class="message">
  <strong>1.</strong> Matrices must be the same size

  <strong>2.</strong> Number of columns of the first matrix must equal the number of rows of the second matrix. The resultant matrix has the same number of rows as the first matrix and the same number of columns as the second matrix. Note that order matters in matrix multiplication.
</p>

Matrix Multiplication dimensions:

$$ \mathbf{A}_{mn}$$ has dimensions of $$m \times n$$

$$ \mathbf{B}_{ij}$$ has dimensions of $$i \times j$$

$$ \mathbf{A}_{mn} \times \mathbf{B}_{ij} = \mathbf{C}_{mj}$$ where $$n = i$$

$$ \mathbf{C}_{mj}$$ has dimensions of $$m \times j$$

## 3. The Unit Vector

The unit vector is a fairly simple concept, it's a vector with a magnitude of 1. This concept applies in any number of dimensions (even infinite). It is often used as a direction vector where the magnitude doesn't matter, they are usually denoted by the a hat on top of their respective letter (e.g. $$\mathbf{\widehat{u}}$$ )

## 4. Matrix Identities, Inverses and Determinants

The identity matrix (written as $$\mathbf{I}$$) is a special matrix that when multiplied with any other matrix gives the matrix itself.

$$\mathbf{A}\cdot \mathbf{I} = \mathbf{A}$$

The inverse of a matrix is used in place of dividing a matrix, instead of dividing matrix $$A$$ by matrix $$B$$ we multiply matrix $$A$$ by the **inverse** of matrix $$B$$ (written as $$B^{-1}$$). it is important to note that inverses only exist for **square** matrices. When a matrix is multiplied by itself and its inverse the result is the identity matrix.

$$\mathbf{AA}^{-1} = \mathbf{I}$$

The inverse is calculated using the determinant, this is a somewhat trivial process that you are asked to do in linear algebra courses but in machine learning you just usually call a function and it does it for you.  Most often you are taught how to deal with finding the inverse of $$2\times2$$ and $$3\times3$$ matrices but not always introduced to a general formula. However determinants can be calculated for any number of dimensions but I will not cover the formula or algorithm in this blog post as it's not really relevant to machine learning, you just need to know it exists.

$$\mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})} \mathrm{adj}(\mathbf{A})$$

Where $$\det(\mathbf{A})$$ is the determinant of $$\mathbf{A}$$ often written as two bars either side of the matrix.

$$\begin{vmatrix} \mathbf{A} \end{vmatrix} = \begin{vmatrix} 1 & 2 \\
                                                             3 & 4 \\
                                                              \end{vmatrix}$$

$$\mathrm{adj}(\mathbf{A})$$ refers to the adjugate of $$\mathbf{A}$$ and is equal to the transpose of the cofactor matrix of A.

$$\mathrm{adj}(\mathbf{A}) = \mathbf{C^T}$$

$$\mathbf{A}^{-1} = \frac{1}{ \begin{vmatrix} \mathbf{A} \end{vmatrix} } \mathbf{C^T}$$

The cofactor matrix can be found using **Cramer's Rule** and is usually taught in High School mathematics, if you are not familiar with it but want to know how it works, do a quick google search.

The transpose of a matrix simply flips the matrix along its diagonal

$$\mathbf{A} = \begin{bmatrix} 1 & 2 \\
                      3 & 4 \\
                      \end{bmatrix}, \mathbf{A^T} = \begin{bmatrix} 1 & 3 \\
                                                                    2 & 4 \\
                                                                    \end{bmatrix}$$

$$(\mathbf{A}^T)^T = \mathbf{A}$$

All of these are easy to find using matlab:

{% highlight matlab %}
% Initial Matrix
A = [1, 2;
     3, 4]

% Inverse of A
B = inv(A)

% Determinant of A
d = det(A)

% Transpose of A
T = A'

{% endhighlight %}

## 5. Dot and Cross Products

The dot and cross product operations are vector operations that are useful for different purposes, the dot product is usually used to find the angle between two vectors and yields a **scalar** result. The cross product yields a **vector** result and is used a lot in physics. You may or may not come across these when reading up about machine learning depending how in depth you study some ML processes. But I would assume most people have a good understanding of how these work

## 6. Transformations

In machine learning and programming in general (e.g. in graphics programming) matrix transformations are used. Essentially you use a predefined matrix to transform a matrix in a given way. Whether that be scaling, rotating etc. Matrix transformations are not to be confused with kernel functions. Often in image processing and machine learning (particularly Convolutional Neural Networks) kernel functions are used instead of matrix transformations, to manipulate data. Kernel functions will be covered in a future blog post.

## 7. Eigenvectors/Eigenvalues

Eigen vectors and Eigen values are a well used part of linear algebra for machine learning, two big areas are for Principal Component Analysis (PCA), used for dimensionality reduction and removing linear correlations and for facial recognition (Eigenfaces!). Both of these topics will be covered in their own blog posts.

The equation that outlines how eigenvectors and eigenvalues work is as follows:

$$ \mathbf{A} \mathbf{x} = \lambda \mathbf{x} $$

Where $$\mathbf{x}$$ is the eigenvector and $$\lambda$$ is the eigenvalue. When multiplying $$\mathbf{A}$$ by the eigenvector it is the same as multiplying the eigenvalue by the eigenvector. The eigenvalue basically tells us what effect multiplying $$\mathbf{A}$$ by the eigenvector has on the eigenvector. There can be multiple eigenvalues and eigenvectors for any given matrix. Rearranging this equation can be useful for many purposes. Applications will be discussed in future posts but this is here as a reference and a reminder of how they work.

That is all! (for now) Again this was intended as a refresher for linear algebra that is used in machine learning and as the first post in my Machine Learning series. If I missed anything I feel is useful I'll add it in future blog posts or just explain it when I use it
