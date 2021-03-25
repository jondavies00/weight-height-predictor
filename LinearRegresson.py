import numpy as np 

def cross_entropy(X, Y, theta):
    guesses = linear_classify(X, theta)
    return -(Y*np.log(guesses) + (1-Y)*np.log(1-guesses))

def total_cross_entropy(X, Y, theta):
    return np.sum(cross_entropy(X, Y, theta))

def d_of_cross_entropy(X, Y, theta, d):
    guesses = linear_classify(X, theta)
    d_wrt_theta = np.zeros((1, d))
    for de in range(d):
        d_wrt_theta[0, de] = np.dot((Y - guesses), np.array(X[:,de]))
    return d_wrt_theta
    

def sigmoid(x):
    return 1/(1+(np.e**-x))

def linear_classify(x, theta):
    return sigmoid(np.dot(x, theta.T)).T #returns 7500 x 1 matrix that has all the x guesses

def gradient_descent(X, Y, f, df, step_size, max_iter, theta_init):
    """
    Performs gradient descent on the given function f, with its gradient df.

    :param f: A function whose input is an x, a column vector, and returns a scalar.
    :param df: A function whose input is an x, a column vector, and returns a column vector representing the gradient of f at x.
    :param x0: An initial value of x, x0, which is a column vector.
    :param step_size: The step size to use in each step
    :param max_iter: The number of iterations to perform

    :return x: the value at the final step
    :return fs: the list of values of f found during all the iterations (including f(x0))
    :return xs: the list of values of x found during all the iterations (including x0)
    """
    fs=[]
    xs=[]
    dimensions = X.shape[1]
    theta = theta_init
    for i in range(max_iter):
        if i == 500:
            print("test")

        fs.append(f(X, Y, theta))

        d_thetas = step_size* df(X, Y, theta, dimensions)

        theta = theta + d_thetas
        xs.append(i)
    return theta, fs, xs