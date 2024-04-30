import numpy as np

from .schemas import ClassifierResults

def cross_entropy(x, y, theta):
    """Gets logistic classification of our x values using theta and returns the cross entropy of each data value"""
    guesses = logistic_classify(x, theta)
    return -(y*np.log(guesses) + (1-y)*np.log(1-guesses))

def total_cross_entropy(x, y, theta):
    """Returns the total amount of cross entropy loss"""
    return np.sum(cross_entropy(x, y, theta))

def d_of_cross_entropy(x, y, theta, d):
    """Calculates the derivative of the cross entropy loss function with respect to theta for each dimension in the data. Returns a column vector of derivatives for each theta."""
    guesses = logistic_classify(x, theta)
    d_wrt_theta = np.zeros((1, d))
    for de in range(d):
        d_wrt_theta[0, de] = np.dot((y - guesses), np.array(x[:,de]))
    return d_wrt_theta
    

def sigmoid(x):
    return 1/(1+(np.e**-x))

def logistic_classify(x, theta):
    """Calculates the dot product of all x values and our thetas, and then applies the sigmoid function. Returns a column vector with all the 'guesses' for the x values."""
    return sigmoid(np.dot(x, theta.T)).T

def gradient_descent(x, y, f, df, theta_init, step_size, max_iter):
    """
    Performs gradient descent on the given function f, with its gradient df.
    :param x: A matrix of features of the dataset
    :param y: A column vector of output labels of the dataset
    :param f: A function whose input is x, a matrix, y and theta, column vectors, and returns a scalar.
    :param df: A function whose input is x, a matrix, y and theta, column vectors, and dimension, the number of dimensions in the data. Returns a column vector representing the gradient of f at x for each dimension.
    :param theta_init: An initial value of theta, which is a column vector.
    :param step_size: The step size to use in each step
    :param max_iter: The number of iterations to perform

    :return x: the value at the final step
    :return fs: the list of values of f found during all the iterations (including f(x0))
    """
    fs=[]
    iters=[]
    dimensions = x.shape[1]
    theta = theta_init
    for i in range(max_iter):
        fs.append(f(x, y, theta))

        d_thetas = step_size* df(x, y, theta, dimensions)

        theta = theta + d_thetas
        iters.append(i)
    return theta, fs