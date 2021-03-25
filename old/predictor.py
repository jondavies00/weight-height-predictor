#Imports
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

# Jonathan Davies' Weight/Height Classifier
# Uses Linear Logistic Classification to classify a given weight or height as male or female.

# TODO:
# 1. Wrangle data into an input and output form
# 2. Define functions for a linear classifier:
#       Apply theta/theta 0 to data -> use sigmoid function to get value between 0 and 1 -> predict +1 or 0, depending on whether guess is >0.5 or <0.5
# 3. Run data to train the classifier
# 4. Test data 

def get_shuffled_csv_data(filename):
    df = pd.read_csv(filename)
    df = df.sample(frac=1)
    return df

def split_data(df, percent_split = int):
    rows = df.shape[0]
    first_rows = int(rows*(percent_split / 100))
    
    last_rows = int(rows - first_rows)
    
    return df.head(first_rows), df.tail(last_rows)

def get_X_and_Y(df):
    Y = df['Gender']
    X = df[['Weight','Height']]
    return X.to_numpy(), Y.to_numpy()

def transform_gender(genders):
    genders[genders=='Male'] = 1
    genders[genders=='Female'] = 0
    return genders

def check_shape(X, Y):
    if X.shape[0] != Y.shape[0]:
        raise Exception("Error: Data is not same shape")
    else:
        print("Same rows!")

def linear_classify(x, theta):
    temp_0 = np.dot(x, theta.T)
    temp = sigmoid(np.dot(x, theta.T))
    temp= temp.flatten()
    return sigmoid(np.dot(x, theta.T)).flatten() #returns 7500 x 1 matrix that has all the x guesses

def sigmoid(x):
    return 1/(1+(np.e**-x))


def negative_log_likelihood_loss(guesses, labels, theta):
    return labels*np.log(linear_classify(guesses, theta)) + (1-labels)*np.log(1-linear_classify(guesses, theta))

def logistic_regression_objective(guesses,labels, theta):
    return np.mean(negative_log_likelihood_loss(guesses, labels, theta))

# def d_of_nll(guesses, labels, theta, feature):
#     sum_deriv = 0
#     for i in range(len(guesses)):
#         sum_deriv += (labels[i] - linear_classify(guesses[i], theta))*guesses[i][feature]
#     return sum_deriv

def d_of_nll(guesses, labels, theta):

    dimensions = theta.shape[1]
    for d in range(dimensions):
        temp_0 = linear_classify(guesses, theta)
        temp_01 = labels
        temp1_1 = temp_01 - temp_0
        temp1 = (labels - linear_classify(guesses, theta))
        temp2 = guesses[:,d].T
        temp = np.dot((labels - linear_classify(guesses, theta)).T, np.array([guesses[:,d]]).T)
        theta+= np.dot((labels - linear_classify(guesses, theta)), np.array([guesses[:,d]]).T)
    return theta

def df_nll(theta ):
    return d_of_nll(training_data_X, training_data_Y, theta)

def f_nll(theta):
    return logistic_regression_objective(training_data_X, training_data_Y, theta)

def gradient_ascent(f, df, theta_init, step_size, max_iter):
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

    fs = []
    xs = []
    thetas = theta_init
    for i in range(max_iter): #for each data example
        fs.append(f(thetas))

        temp = step_size*df(thetas)
        thetas = step_size*df(thetas) #modify that feature by using the derivative of log likelihood
        xs.append(thetas.flatten())
        if i % 10 == 0:
            print(i, thetas)

    return thetas, fs, xs

def test_theta(theta, test_data_input, test_data_labels):
    print("Guessed: " , linear_classify(test_data_input, theta), " with labels ", test_data_labels)


#split data into 75/25 for training/testing
seventy_five_split, twenty_five_split = split_data(get_shuffled_csv_data('weight-height.csv'), 75)



#convert training data to numpy arrays
training_data_X, training_data_Y = get_X_and_Y(seventy_five_split)

#transform genders to binary zero or one 
training_data_Y = transform_gender(training_data_Y)

test_data_X, test_data_Y = get_X_and_Y(twenty_five_split)
test_data_Y = transform_gender(test_data_Y)
test_data_X = np.flip(np.append(np.ones((test_data_X.shape[0], 1)), test_data_X, axis=1), axis=1)

#ensure arrays are of same shape
print("Checking shape...")
check_shape(training_data_X, training_data_Y)

plt.scatter(test_data_X[:,0], test_data_X[:,1])

plt.show()

training_data_X = np.flip(np.append(np.ones((training_data_X.shape[0], 1)), training_data_X, axis=1), axis=1)

theta = np.array([[0.001,0.001,0.001]])

last_theta, fs, xs = gradient_ascent(f_nll, df_nll, theta, 0.000001, 1000)

print(last_theta, training_data_X, training_data_Y)
print(f_nll(last_theta))
test_theta(theta, test_data_X, test_data_Y)