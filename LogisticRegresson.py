import numpy as np 

def cross_entropy(X, Y, theta):
    """Gets logistic classification of our X values using theta and returns the cross entropy of each data value"""
    guesses = logistic_classify(X, theta)
    return -(Y*np.log(guesses) + (1-Y)*np.log(1-guesses))

def total_cross_entropy(X, Y, theta):
    """Returns the total amount of cross entropy loss"""
    return np.sum(cross_entropy(X, Y, theta))

def d_of_cross_entropy(X, Y, theta, d):
    """Calculates the derivative of the cross entropy loss function with respect to theta for each dimension in the data. Returns a column vector of derivatives for each theta."""
    guesses = logistic_classify(X, theta)
    d_wrt_theta = np.zeros((1, d))
    for de in range(d):
        d_wrt_theta[0, de] = np.dot((Y - guesses), np.array(X[:,de]))
    return d_wrt_theta
    

def sigmoid(x):
    return 1/(1+(np.e**-x))

def logistic_classify(X, theta):
    """Calculates the dot product of all x values and our thetas, and then applies the sigmoid function. Returns a column vector with all the 'guesses' for the x values."""
    return sigmoid(np.dot(X, theta.T)).T

def gradient_descent(X, Y, f, df, theta_init, step_size, max_iter):
    """
    Performs gradient descent on the given function f, with its gradient df.
    :param X: A matrix of features of the dataset
    :param Y: A column vector of output labels of the dataset
    :param f: A function whose input is x, a matrix, y and theta, column vectors, and returns a scalar.
    :param df: A function whose input is x, a matrix, y and theta, column vectors, and dimension, the number of dimensions in the data. Returns a column vector representing the gradient of f at x for each dimension.
    :param theta_init: An initial value of theta, which is a column vector.
    :param step_size: The step size to use in each step
    :param max_iter: The number of iterations to perform

    :return x: the value at the final step
    :return fs: the list of values of f found during all the iterations (including f(x0))
    :return iters: the list of values of iterations
    """
    fs=[]
    iters=[]
    dimensions = X.shape[1]
    theta = theta_init
    for i in range(max_iter):
        if i == 500:
            print("test")

        fs.append(f(X, Y, theta))

        d_thetas = step_size* df(X, Y, theta, dimensions)

        theta = theta + d_thetas
        iters.append(i)
    return theta, fs, iters