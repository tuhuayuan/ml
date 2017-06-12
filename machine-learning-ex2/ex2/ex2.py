#-*- coding: utf-8 -*-
"""编程作业"""
#%%
import numpy as np

data1 = np.loadtxt('ex1data1.txt', delimiter=',')
m = data1.shape[0]
ones = np.ones((m, 1))

data1 = np.insert(data1, [0], ones, axis=1)
X = data1[:, :2]
y = data1[:, [2]]


class LineRegression(object):
    """线性回归"""

    def __init__(self):
        pass


def cost(x, y, theta):
    """ 价值函数 """
    m = x.shape[0]
    J = np.subtract(np.dot(x, theta), y)
    J = np.power(J, 2)
    J = np.sum(J)/(2*m)
    return J

def descent(x, y, theta, alpha, num_iters):
    '''
    梯度下降函数
    '''
    feature_count = x.shape[1]
    m = x.shape[0]
    for i in range(0, num_iters):
        t = np.zeros((feature_count,1))
        for feature_index in range(0, feature_count):
            featurex = x[:,[feature_index]]
            derivative = np.sum(np.subtract(np.dot(x, theta), y) * featurex)/m
            t[feature_index] = theta[feature_index] - alpha*derivative
        theta = t
    return theta

#%%
theta = descent(X, y, np.zeros((2, 1)), 0.01, 1500)
print(theta)
