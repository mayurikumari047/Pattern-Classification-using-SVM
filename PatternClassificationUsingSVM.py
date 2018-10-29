# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:43:06 2017

@author: mayur
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import cvxopt

def generate_random_number_matrix_in_range(n, m, r_start, r_end):
    random_number_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            random_number_matrix[i][j] = random.uniform(r_start, r_end)
    return random_number_matrix


def kernel(x, y):
    return (1 + np.dot(x, y)) ** 5


N = 100
X = list()
d = list()
X_class1 = list()
X_class0 = list()
X1 = np.zeros((N, 1))
X2 = np.zeros((N, 1))

plt.title("Classes without SVM")

for i in range(0, N):
    temp = np.random.uniform(0, 1, 2)
    X.append(temp)

    X1[i][0] = temp[0]
    X2[i][0] = temp[1]

    if (X2[i][0] < ((1 / 5) * np.sin(10 * X1[i][0]) + 0.3)) or (
    (math.pow((X2[i][0] - 0.8), 2) + math.pow((X1[i][0] - 0.5), 2))) < math.pow(0.15, 2):
        d.append(1.0)
        X_class1.append(temp)
        plt.plot(X1[i][0], X2[i][0], 'rx')
    else:
        d.append(-1.0)
        X_class0.append(temp)
        plt.plot(X1[i][0], X2[i][0],'bd')

X = np.array(X)

plt.ylabel('xi2')
plt.xlabel('xi1')
plt.show()


K = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i][j] = kernel(X1[i], X2[j])


P = np.zeros((N, N))
for i in range(N):
    for j in range(N):
    	P[i][j] = ((K[i][j]) * d[i] * d[j])
        
P = cvxopt.matrix(P)
G = np.identity(N)
G = cvxopt.matrix(G * -1)

A = cvxopt.matrix(np.matrix(np.array(d)))
b = cvxopt.matrix(0.0)
q = cvxopt.matrix(np.ones(N) * -1)
h = cvxopt.matrix(np.zeros((N, 1)))

sol = cvxopt.solvers.qp(P,q,G,h,A,b)

aplhas_matrix = sol['x']
alpha_list = np.ravel(aplhas_matrix)

print(alpha_list)

support_vectors_x = []
support_vectors_y = []
for i in range(0,len(alpha_list)):
    if (alpha_list[i] > 95):
        temp = X[i]
        print (i)
        support_vectors_x.append(temp[0])
        support_vectors_y.append(temp[1])

print("Support vectors length:", len(support_vectors_x))

sv = X[30]
sv_x = sv[0]
sv_y = sv[1]
theta = 0.0
sum = 0.0
for i in range(0, len(X)):
    temp = alpha_list[i] * d[i] * kernel(X[i],sv)
    sum = sum + temp
print (sum)
theta = d[30] - sum
print (theta)

random_x1 = np.linspace(0,1,1000)
random_y1 = np.linspace(0,1,1000)
random_x = []

for i in range(0,1000):
    for j in range(0,1000):
        random_x.append([random_x1[i],random_y1[j]])
print (len(random_x))


def calculate_discriminant():
    sum_discriminant = 0.0
    for i in range(0, len(x_i)):
        temp = alpha_list[i] * d[i] * kernel(X[i], random_X)
        sum_discriminant = sum_discriminant + temp

    sum_discriminant = sum_discriminant + theta
    return sum_discriminant


Hplus = []
Hminus = []
H = []
for i in range(0, len(random_x)):
    temp = calculate_discriminant(random_x[i])
    if temp < 0.1 and temp > -0.1:
        H.append(random_x[i])

    if temp < 1.1 and temp > 0.9:
        Hplus.append(random_x[i])

    if temp < -0.9 and temp > -1.1:
        Hminus.append(random_x[i])

print(len(H))
print(len(Hplus))
print(len(Hminus))

fig, ax = plt.subplots(figsize=(10,10))
plt.title("Class separation using SVM")
plt.plot(x_class_0,y_class_0, 'rx')
plt.plot(x_class_1,y_class_1, 'bd')
plt.plot(*zip(*Hplus), c = 'red',s = 1, label = 'Hyperplane 1')
plt.plot(*zip(*H), c = 'black', s = 1,label = 'Margin')
plt.plot(*zip(*Hminus), c = 'blue',s = 1, label = 'Hyperplane -1')
plt.ylabel('xi2')
plt.xlabel('xi1')
plt.show()
