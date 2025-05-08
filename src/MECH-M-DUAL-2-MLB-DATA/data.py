import numpy as np
import logging
import etl


def load_cats_vs_dogs():
    cats_w = etl.load("catData_w.mat")
    dogs_w = etl.load("dogData_w.mat")

    X_train = np.concatenate((cats_w[:60, :], dogs_w[:60, :]))
    y_train = np.repeat(np.array([1, -1]), 60)
    X_test = np.concatenate((cats_w[60:80, :], dogs_w[60:80, :]))
    y_test = np.repeat(np.array([1, -1]), 20)
    logging.debug("Loaded the data with Split of 60 to 20 per category.")

    return X_train, y_train, X_test, y_test
