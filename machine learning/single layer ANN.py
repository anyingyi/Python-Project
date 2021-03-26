# -*- coding: utf-8 -*-
"""
single layer ANN: gradient descent method for calculate wi

Created on Sat Sep 19 15:12:16 2020

@author: Anyingyi
"""

import numpy as np






def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))




class NeuralNetwork:

    def __init__(self):
        # hyper parameters
        np.random.seed(42)
        self.weights = np.random.rand(200, 1)
        self.bias = np.random.rand(1)
        self.lr = 0.05
        #print(weights)

        #aa = np.dot(feature_set, self.weights)

        #xx = aa + self.bias

    # train
    def fit(self,feature_set,labels):
        for epoch in range(20000):
            inputs = feature_set

            # feedforward step1
            XW = np.dot(feature_set, self.weights) + self.bias

            # feedforward step2
            z = sigmoid(XW)

            # backpropagation step 1
            error = z - labels

            if epoch % 100 == 0:
                #print(error.sum())
                pass

            # backpropagation step 2
            dcost_dpred = error
            dpred_dz = sigmoid_der(z)

            z_delta = dcost_dpred * dpred_dz

            inputs = feature_set.T
            self.weights -= self.lr * np.dot(inputs, z_delta)

            for num in z_delta:
                self.bias -= self.lr * num

    # pridict
    def predict(self,x):
        result = sigmoid(np.dot(x, self.weights) + self.bias)
        return result


nn = NeuralNetwork()
X = np.linspace(-np.pi,np.pi,200)
y = np.sin(X)
nn.fit(X, y)
for e in np.linspace(-np.pi,np.pi,20):
    print(e)
    #print("{}:{}\n".format(e,nn.predict(e)))


