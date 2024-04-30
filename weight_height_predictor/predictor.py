from dataclasses import dataclass
import random
from matplotlib import pyplot as plt
import numpy as np

from .data_transform import check_shape, round_classification, split_data, get_shuffled_csv_data, get_X_and_Y, transform_gender, append_ones, get_missclassified
from .logistic_regressor import logistic_classify, sigmoid, total_cross_entropy, d_of_cross_entropy, gradient_descent

from .schemas import ClassifierResults

def guess_user_input(th):
    """
    Runs a loop that allows the user to enter weight and height to see if their sex can be correctly predicted
    """
    exit_input = ""
    while exit_input != "e":
        height = input("Enter height (inches): ")
        weight = input("Enter weight (pounds): ")
        h_w = np.zeros((1, 3))
        h_w[0,0] = height
        h_w[0,1] = weight
        h_w[0,2] = 1
        classified = logistic_classify(h_w, th)
        if round_classification(classified) == 1:
            print("I guess you are a male")
        else:
            print("I guess you are a female")
        exit_input = input("Enter 'e' to exit...")

def plot_error(errors: list[int]):
     # Create a plot to show how cross entropy error gets smaller over iterations
    fig, ax = plt.subplots()

    ax.set_title("Gradient Descent of cross entropy error")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cross Entropy Error")
    ax.scatter(range(len(errors)), errors)
    
    plt.savefig("error_plot")

def classify_data():
    # Initialise thetas as randomly generated values. They are put through the sigmoid function so they are small enough that cross error entropy can be calculated
    # Note: the model still works with any theta values, but may need more iterations
    theta_init = np.array([[sigmoid(random.randint(0, 100)), sigmoid(random.randint(0, 100)), sigmoid(random.randint(0, 100))]])

    # Shuffle the data, and split it 1:3 so we have a training set and a test test
    # Note: data must be shuffled as male and female data are separated in the dataset
    seventy_five_split, twenty_five_split = split_data(get_shuffled_csv_data('data/weight-height.csv'), 75)

    # Seperate the data features and labels from the data
    training_data_X, training_data_Y = get_X_and_Y(seventy_five_split)

    # Append a dimension of ones to the features, to act as theta_0 (a bias) and allow the model to fit properly
    training_data_X = np.flip(np.append(np.ones((training_data_X.shape[0], 1)), training_data_X, axis=1), axis=1)

    # Transform male/female genders to a binary one/zero respectively
    training_data_Y = transform_gender(training_data_Y)

    # Get our test data ready from the 25% split
    test_data_X, test_data_Y = get_X_and_Y(twenty_five_split)
    test_data_Y = transform_gender(test_data_Y)
    test_data_X = append_ones(test_data_X)

    # Ensure training data and test data features and labels are of same shape.
    print("Checking shape...")
    check_shape(training_data_X, training_data_Y)
    check_shape(test_data_X, test_data_Y)

    # Initialise parameters for gradient descent
    step_size = 0.0000001 # step size seems to work with any value, but cross entropy loss cannot be calculated without a very small one
    max_iter = 500 # 500 seems to be the best number of iterations

    #Perform gradient descent with our parameters
    thetas, errors = gradient_descent(training_data_X, training_data_Y, total_cross_entropy, d_of_cross_entropy, theta_init, step_size, max_iter)

    print("Found thetas:", thetas)

    # Classify our test data using our model
    classified_data = logistic_classify(test_data_X, thetas)
    # Round the values so they are binary
    classified_data = round_classification(classified_data)

    # Calculate accuracy of model
    total_missclassified = get_missclassified(test_data_Y, classified_data)
    total_data = len(test_data_Y)
    print("Total missclassfied data =", total_missclassified, ", Total data =", total_data)
    print("Accuracy =" , str(((total_data - total_missclassified)/ total_data) * 100) + "%")

    return ClassifierResults(errors=errors, thetas=thetas)