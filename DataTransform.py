import pandas as pd
import numpy as np

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

def round_classification(classified_data):
    classified_data[classified_data > 0.5] = 1
    classified_data[classified_data < 0.5] = 0
    return classified_data

def check_shape(X, Y):
    if X.shape[0] != Y.shape[0]:
        raise Exception("Error: Data is not same shape")
    else:
        print("Same rows!")

def append_ones(X):
    """
    Appends a dimension of ones to given numpy array

    :param X: a numpy array
    :return: a numpy array with an added dimension (column) of ones
    """
    return np.flip(np.append(np.ones((X.shape[0], 1)), X, axis=1), axis=1)

def get_missclassified(labels, classified_data):
    return np.sum(np.abs(labels - classified_data))