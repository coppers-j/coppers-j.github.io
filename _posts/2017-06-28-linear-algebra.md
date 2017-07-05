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
8. Vector Spaces
9. Real World Applications


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

$$ A =  \begin{matrix}
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

  <strong>2.</strong> Number of columns of the first matrix must equal the number of rows of the second matrix. The resultant matrix has the same number of rows as the first matrix and the same number of columns as the second matrix.
</p>

Matrix Multiplication dimensions:

$$ A_{mn}$$ has dimensions of $$m \times n$$

$$ B_{ij}$$ has dimensions of $$i \times j$$

$$A_{mn} \times B_{ij} = C_{mj}$$ where $$n = i$$

$$C_{mj}$$ has dimensions of $$m \times j$$

## 3. The Unit Vector

The unit vector is a fairly simple concept, it's a vector with a magnitude of 1. This concept applies in any number of dimensions (even infinite). It is often used as a direction vector where the magnitude doesn't matter, they are usually denoted by the a hat on top of their respective letter (e.g. $$\mathbf{\widehat{u}}$$ )

## 4. Matrix Identities, Inverses and Determinants


## 5. Dot and Cross Products


## 6. Transformations


## 7. Eigenvectors/Eigenvalues


## 8. Vector Spaces


## 9. Real World Applications
