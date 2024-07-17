import numpy as np
import matplotlib.pyplot as plt

def linear_regression(X, Y, learning_rate, iterations, num_features):
    m = X.size
    Z = np.hstack((np.ones((m,1)), X))
    theta = np.zeros((num_features+1,1))
    cost_list = []

    for i in range(iterations):
        y_hat = np.dot(Z, theta)
        cost = 1/(2*m)*np.sum(np.square(y_hat-Y))
        d_theta = (1/m)*np.dot(Z.T, y_hat - Y)
        theta = theta - learning_rate*d_theta
        cost_list.append(cost)

    return theta, cost_list