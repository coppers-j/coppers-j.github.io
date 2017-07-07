---
layout: post
title: Supervised vs. Unsupervised Learning
categories: ML
---

The two main categories of machine learning that most algorithms and models fall into are supervised and unsupervised learning. I plan on covering both of these topics individually in more detail but this seemed like a relevant precursor. Along the way in this series of posts I'll relate back to supervised and unsupervised techniques and when to use them.

Supervised learning is where the target or desired output of a system is known and stored in the training set. For example, if we wanted to train a simple regression model to find the height of someone given the weight, we would have a set of data containing the heights and weights of many different people and use that to create a polynomial model and determine a function of weight to height.

Unsupervised learning can often be referred to as feature extraction, clustering is a simple example of this. With a simple 2d system it doesn't matter what each dimension represents but finding cluster centers and then creating a model or discriminant to determine which cluster center each data point belongs is still machine learning. This process is often used in online shopping systems to classify a *type* or *group* of shoppers and show them targeted advertisement for what other people in the same grouping are buying.

The first few machine learning techniques I will cover will be supervised but unsupervised learning will follow. Just a short blog post for today but the next post will be on probabilistic classification.
