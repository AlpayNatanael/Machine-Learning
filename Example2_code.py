import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import Perc_code as p

# activitaion function
def ActivFunc(v, B = False, H = True):
    """
    Function to calculate the activation function result.
    B - Binomial
    H - Hyperbolic tangent function
    """
    if(H):
        x = (1  - np.exp(-2*v))/(1 + np.exp(-2*v))

    if(B):
        if v < 0:
            x = 0
        else:
            x = 1
    return x

# loading the data
# %matplotlib inline
def load_data_set(show = True, line = False, weights = []):
    """ Show- true graph, else no grapth"""
    #https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

    # return the data and the target
    X,y = load_iris(return_X_y=True)

    if (show == True):
        # plt.scatter(x, y,...)
        # first fifty enteries
        plt.scatter(X[:50, 0], X[:50, 1],
                    color='green', marker='x', label='setosa')

        # last thifty entires
        plt.scatter(X[50:100, 0], X[50:100, 1],
                    color='red', marker='o', label='versicolor')
        plt.xlabel('sepal length')
        plt.ylabel('petal length')
        plt.legend(loc='upper right')
        plt.show()

    if (line == True):
        plt.scatter(X[:50, 0], X[:50, 1],
                    color='green', marker='x', label='setosa')

        # last thifty entires
        plt.scatter(X[50:100, 0], X[50:100, 1],
                    color='red', marker='o', label='versicolor')
        plt.xlabel('sepal length')
        plt.ylabel('petal length')
        plt.legend(loc='upper right')
        plt.show()
        line_plt = []
        # Here i am calculating slope and intercept with given three weights
        for i in np.linspace(np.amin(X[:50,:1]),np.amax(X[:50,:1])):
            slope = -(weights[0]/weights[2])/(weights[0]/weights[1])
            intercept = -weights[0]/weights[2]

            #y =mx+c, m is slope and c is intercept
            y = (slope*i) + intercept
            plt.plot(i, y,'ko')
            #line_plt.append([i,y,'ko'])

        #plt.scatter(line_plt[:,0], line_plt[:,1],
        #            color='red', marker='o', label='versicolor')
        #plt.show()
    else:
        return X,y

class Perceptron(object):
    """
    Class for that describes an object perceptron, that represents one hidden
    layer in a multilayer Perceptron system
    self - the thing.
    learning_rate - the wieght, eta.
    n_iter - number of iteration
    random_state - to see later.
    """

    # constractor
    def __init__(self, learning_rate=0.01, n_iter=100, random_state=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Note: In the book inpt is Y in code input is X. In Book target is D, in code target is Y.
        Maps the traings data X and the target Y, then updates and return the perceptron.
        """
        rand = np.random.RandomState(self.random_state)
        # create a vector of weights of of size 'size of x + 1' with first entery as bias.
        self.weights = rand.normal(loc=0.0, scale=0.01, size=1 +  X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for x, target in zip(X, y):
                # local gradient: delta_j
                update = self.learning_rate * (target - self.predict(x))

                # Weight correction. i.e. we update the weights as in the back-propogation algorithm.
                self.weights[1:] += update * x

                # We update the bias
                self.weights[0] += update

                # if the update is not zero, then we have an error as we didnot predict correctly.
                errors += int(update != 0.0)

                # add the error as a negative value
                self.errors_.append(errors)
                #self.errors_.insert(0,errors)
            #plot_data(self, X,y, self.weights[1:])
            return self

    def net_input(self, X):
        """"we calculate the neuron biased induced field and call the activation function"""
        # \vec x \dot \vec w + bias
        v = np.dot(X, self.weights[1:]) + self.weights[0] #bias is the first entry in the weight vector
        v = ActivFunc(v)
        # print(z)
        return v

    # method documentation https://numpy.org/doc/stable/reference/generated/numpy.where.html
    def predict(self, X):
        """ Note: This part is still in understanding stage.

        Prediction is made based upon the summation result (?),
        returns an array of 0 1 based on the activation is positive or not.
        """
        # if get back a vector of 1,0 where 1 is for positive function from the activision function
        return np.where(self.net_input(X) >= 0, 1, 0)# possible last entry is -1

    def compute(self):
        # plotting the error-energy function \xi_avg
        X,y = load_data_set(show = False)
        self.fit(X,y)
        plt.plot(range(1, len(self.errors_) + 1), self.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
        plt.show()






    ######   aux functions, below code is not working yet ######


    #https://stats.stackexchange.com/questions/71335/decision-boundary-plot-for-a-perceptron
    def plot_data(self):#,inputs,targets,weights):
        inputs,targets = load_data_set(False)
        #print(inputs)
        #print()
        #print(inputs[:50])

        #print(targets)
        #print()
        #print(targets[:50])


        inputs = inputs[:50]
        targets = targets[:50]

        weights = self.weights[1:]

        # fig config
        #plt.figure(figsize=(10,6))

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
        plt.show()
