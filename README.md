# Pattern-Classification-using-SVM
Design and implementation of Support Vector Machine from scratch without any use of machine learning libraries for Pattern classification.

Support vector machines are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. 

Given a set of training examples, each marked as belonging to one or the other of two categories, a SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier. A SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.

When data are not labeled, supervised learning is not possible, and an unsupervised learning approach is required, which attempts to find natural clustering of the data to groups, and then map new data to these formed groups. The clustering algorithm which provides an improvement to the support vector machines is called support vector clustering.

I chose the below mentioned Kernel function and this gave the better classification as compared to linear.

#### Input details, Kernel function and number of support vectors:

Input length: 100

Input range: 0 to 1

Kernel function used: (1 + x1 * x2) ^ 5

Number of support vectors: 12

#### Color legends used in graphs:

For input x1: red x

For input x2: blue diagonal

For Hyperplane, H+:Red

For Hyperplane, H-:Blue

For Optimal hyperplane, H:Black


