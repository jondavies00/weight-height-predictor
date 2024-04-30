## Weight/Height Gender Predictor

This logistic regression classifier loads and trains on a dataset of weights, heights and gender. It then allows for user input to guess the gender from some weight & height. It also generates a plot of error during training as a side effect.

When ran, the dataset will automatically be loaded and trained on and a model will be fit. THis is done using gradient descent and the cross entropy loss function.

It achieves around 92% accuracy, which is on par with a model from the sklearn module for this dataset.

### Prerequiites:

- Python 3.11
- Poetry

### Usage

To use,
1. Clone the repo
2. Run `poetry install` to install dependencies
3. Run `poetry run python -m weight_and_height_predictor`. 
4. Input any weight/height you want the classifier to guess
