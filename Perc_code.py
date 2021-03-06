import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import seaborn


#Source: https://towardsdatascience.com/perceptron-and-its-implementation-in-python-f87d6c7aa428

# out activitaion function
#def ActivFunc(v):
#    return (1 - np.exp(-2*v))/(1 + np.exp(-2*v))

def ActivFunc(v):
    x = (1 - np.exp(-2*v))/(1 + np.exp(-2*v))
    if x < 0:
        x = 0
    else:
        x = 1
    return x
    



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
            y = np.dot(w, x.transpose()) # we take the transpose to do the dot product
            delta = (label.item(0,0) - ActivFunc(y))

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



########################################################


# aux function for ploting, not working yet

#https://stats.stackexchange.com/questions/71335/decision-boundary-plot-for-a-perceptron
def plot_data(inputs,targets,weights):
    # fig config
    plt.figure(figsize=(10,6))
    plt.grid(True)

    #plot input samples(2D data points) and i have two classes.
    #one is +1 and second one is -1, so it red color for +1 and blue color for -1
    for input,target in zip(inputs,targets):
        plt.plot(input[0],input[1],'ro' if (target == 1.0) else 'bo')

    # Here i am calculating slope and intercept with given three weights
    for i in np.linspace(np.amin(inputs[:,:1]),np.amax(inputs[:,:1])):
        slope = -(weights[0]/weights[2])/(weights[0]/weights[1])
        intercept = -weights[0]/weights[2]

        #y =mx+c, m is slope and c is intercept
        y = (slope*i) + intercept
        plt.plot(i, y,'ko')
