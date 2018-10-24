import numpy as np

def split_dataset(X, y, column, t, return_x=True):
    assert isinstance(X, np.ndarray), "X must be ndarray type"
    assert isinstance(y, np.ndarray), "y must be ndarray type"

    try:
        y_left = y[X[:, column] < t]
        y_right = y[X[:, column] >= t]
    except TypeError:
        raise TypeError("threshold must be a number")

    if not return_x:
        return y_left, y_right

    X_left = X[X[:, column] < t]
    X_right = X[X[:, column] >= t]

    return X_left, X_right, y_left, y_right

def split_by_column(column, y, t):
    """split label vector based on column by threshold"""
    assert isinstance(column, np.ndarray), "X must be ndarray type"
    assert isinstance(y, np.ndarray), "y must be ndarray type"

    assert column.shape[0] == y.shape[0]

    try:
        y_left = y[column < t]
        y_right = y[column >= t]
    except TypeError:
        raise TypeError("threshold must be a number")

    return y_left, y_right
