import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import seaborn

# out activitaion function
def ActivFunc(v):
    return (1 - np.exp(-2*v))/(1 + np.exp(-2*v))

# our data, with deafult data set being the iris data set
def load_data(viz = True):
    URL_='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(URL_, header = None)


    # make the dataset linearly separable
    # Would be unique per data set
    data = data[:100]
    data[4] = np.where(data.iloc[:, -1]=='Iris-setosa', 0, 1)
    data = np.asmatrix(data, dtype = 'float64')

    #data visualisation
    if (viz == True):
        # print(data)

        plt.scatter(np.array(data[:50,0]), np.array(data[:50,2]), marker='o', label='setosa')
        plt.scatter(np.array(data[50:,0]), np.array(data[50:,2]), marker='x', label='versicolor')
        plt.xlabel('petal length')
        plt.ylabel('sepal length')
        plt.legend()
        plt.show()
    else:
        return data


# Perceptron implementation
def perceptron(data, num_iter):
    features = data[:, :-1]
    labels = data[:, -1]

    # set weights to zero
    w = np.zeros(shape=(1, features.shape[1]+1))

    misclassified_ = []


    # the iterations
    for epoch in range(num_iter):
        misclassified = 0
        for x, label in zip(features, labels):
            x = np.insert(x,0,1)
            y = np.dot(w, x.transpose())

            # target is the result of the activation function
            if ActivFunc(y)>0:
                target = 1.0
            else:
                target = 0.0

            delta = (label.item(0,0) - target)

            if(delta): # misclassified
                misclassified += 1
                w += (delta * x)

        misclassified_.append(misclassified)
    return (w, misclassified_)



def ComputePreceptron(D, N = 10):
    """
    N - Number of iteration
    D - data
    """
    data = D
    num_iter = N
    w, misclassified_ = perceptron(data, num_iter)

    epochs = np.arange(1, num_iter+1)
    plt.plot(epochs, misclassified_)
    plt.xlabel('iterations')
    plt.ylabel('misclassified')
    plt.show()
