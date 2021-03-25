import DataTransform as dt
import LinearRegresson as lr
import numpy as np
import matplotlib.pyplot as plt

def input_data():
    exit_input = ""
    while exit_input != "e":
        height = input("Enter height (inches): ")
        weight = input("Enter weight (pounds): ")
        h_w = np.zeros((1, 3))
        h_w[0,0] = height
        h_w[0,1] = weight
        h_w[0,2] = 1
        classified = lr.linear_classify(h_w, th)
        if dt.round_classification(classified) == 1:
            print("I guess you are a male")
        else:
            print("I guess you are a female")
        exit_input = input("Enter 'e' to exit...")

theta_init = np.array([[0.05, 0.85, 0.3]])

seventy_five_split, twenty_five_split = dt.split_data(dt.get_shuffled_csv_data('weight-height.csv'), 75)

training_data_X, training_data_Y = dt.get_X_and_Y(seventy_five_split)

training_data_X = np.flip(np.append(np.ones((training_data_X.shape[0], 1)), training_data_X, axis=1), axis=1)

#transform genders to binary zero or one 
training_data_Y = dt.transform_gender(training_data_Y)

test_data_X, test_data_Y = dt.get_X_and_Y(twenty_five_split)
test_data_Y = dt.transform_gender(test_data_Y)
test_data_X = dt.append_ones(test_data_X)

#ensure arrays are of same shape
print("Checking shape...")
dt.check_shape(training_data_X, training_data_Y)

th, fs, xs = lr.gradient_descent(training_data_X, training_data_Y, lr.total_cross_entropy, lr.d_of_cross_entropy, 0.0000001, 200, theta_init)

print("Found thetas:", th)


classified_data = lr.linear_classify(test_data_X, th)
classified_data = dt.round_classification(classified_data)

total_missclassified = dt.get_missclassified(test_data_Y, classified_data)
total_data = len(test_data_Y)
print("Total missclassfied data =", total_missclassified, ", Total data =", total_data)
print("Accuracy =" , str(((total_data - total_missclassified)/ total_data) * 100) + "%")

fig, ax = plt.subplots()

ax.set_title("Gradient Descent of cross entropy error")
ax.set_xlabel("Iterations")
ax.set_ylabel("Cross Entropy Error")
ax.plot(xs, fs)
plt.show()

input_data()