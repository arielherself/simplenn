import simplenn
import MNIST
import numpy as np
from time import sleep
from copy import deepcopy

sigmoid = simplenn.Sigmoid()
cross_entropy = simplenn.CrossEntropy()

x, y = MNIST.load_data('./data/mnist_train.csv', rescale=True)
m, n = x.shape
units = np.array([25, 15, 1])
model = simplenn.Model(units, n, sigmoid, cross_entropy)
# model.evaluate(x, y)
# sleep(5)
print('\n============Train===========')
model.train_2(x, y, 20, 0.001, 1)
print('\n=====Evaluate(training)=====')
model.evaluate(x, y)
print('\n=======Evaluate(test)=======')
x, y = MNIST.load_data('./data/mnist_test.csv', rescale=True)
model.evaluate(x, y)

# x, y = MNIST.load_data('./data/mnist_train.csv', max_size=100)
# m, n = x.shape
# units = np.array([25, 15, 1])
# model = simplenn.Model(units, n, sigmoid, cross_entropy)
# model1 = deepcopy(model)
# model1.train_ng(x, y, 20, 0.6)
# model1.evaluate(x, y)

# x = np.array([[0, 0],
#               [1, 1]])
# y = np.array([0, 1])
#
# units = np.array([3, 1])
# model = simplenn.Model(units, 2, sigmoid, cross_entropy)
# model.train_ng(x, y, 500, 0.6)
# model.evaluate(x, y)