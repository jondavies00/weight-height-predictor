import pandas as pd
import numpy as np

def get_shuffled_csv_data(filename):
    """Shuffles data from a csv file and returns a dataframe of the newly shuffled data."""
    df = pd.read_csv(filename)
    df = df.sample(frac=1)
    return df

def split_data(df, percent_split = int):
    """
    Splits data of specified percentage
    :param df: a dataframe containing at least the denominator of rows of the fractional percentage
    :param percent_split: the percentage to split the data by as an integer
    :return: two dataframes containing all the data, split by the percentage, with the first containing the percentage specified and the second containing the rest
    """
    rows = df.shape[0]
    first_rows = int(rows*(percent_split / 100))
    
    last_rows = int(rows - first_rows)
    
    return df.head(first_rows), df.tail(last_rows)

def get_X_and_Y(df):
    """Transforms gender, weight, and height into two numpy arrays of features and labels."""
    y = df['Gender']
    x = df[['Weight','Height']]
    return x.to_numpy(), y.to_numpy()

def transform_gender(genders):
    """Transforms gender into 1 if it is male, and 0 if female. Returns the numpy array of these values."""
    genders[genders=='Male'] = 1
    genders[genders=='Female'] = 0
    return genders

def round_classification(classified_data):
    """Rounds classification of data to 1 if equal to or above 0.5 and 0 if below 0.5. Returns the newly classified data."""
    classified_data[classified_data >= 0.5] = 1
    classified_data[classified_data < 0.5] = 0
    return classified_data

def check_shape(x, y):
    """Ensures data from two numpy arrays has the same amount of rows."""
    if x.shape[0] != y.shape[0]:
        raise Exception("Error: Different amounts of data.")
    else:
        print("Same rows!")

def append_ones(x):
    """
    Appends a dimension of ones to given numpy array

    :param x: a numpy array
    :return: a numpy array with an added dimension (column) of ones
    """
    return np.flip(np.append(np.ones((x.shape[0], 1)), x, axis=1), axis=1)

def get_missclassified(labels, classified_data):
    """Returns the sum of the absolute differences between the correct labels, and the classified data."""
    return np.sum(np.abs(labels - classified_data))