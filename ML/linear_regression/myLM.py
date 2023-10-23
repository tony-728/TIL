import numpy as np
import pandas as pd


class MultipleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 1

    def forward(self, X):
        """

        Parameters
        ----------
        X : array
            input data
            (number of input data, number of feature)

        Returns
        -------
        _type_
            _description_
        """
        return X @ self.coef_ + self.intercept_  # (N,1)

    def get_gradient(self, X, Y):
        w_gradient = list()

        # 가중치 미분
        for i in range(X.shape[1]):  # 2
            gradient = (self.forward(X) - Y.reshape(len(X), -1)) * X[:, i].reshape(
                len(X), -1
            )
            w_gradient.append(gradient.sum() / len(X))

        w_gradient = np.array(w_gradient).reshape(X.shape[1], -1)

        # 편향 미분
        b_gradient = (self.forward(X) - Y.reshape(len(X), -1)).sum() / len(X)

        return w_gradient, b_gradient

    def fit(self, X, y, learning_rate=0.001, epoch=10000):
        self.coef_ = np.ones((X.shape[1], 1))

        for _ in range(epoch):
            w_gradient, b_gradient = self.get_gradient(X, y)
            self.coef_ = self.coef_ - (learning_rate * w_gradient)
            self.intercept_ = self.intercept_ - (learning_rate * b_gradient)


def minmax_normalize(X):
    for c in X.columns:
        minimum = X[c].min()
        maximum = X[c].max()

        X.loc[:, c] = (X.loc[:, c] - minimum) / (maximum - minimum)

    return X


def standardize(X):
    for c in X.columns:
        m = X[c].mean()
        s = X[c].std()

        X.loc[:, c] = (X.loc[:, c] - m) / s

    return X


column = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRAT",
    "B",
    "LSTAT",
    "MEDV",
]
data = pd.read_csv("ML\\linear_regression\\housing.csv", names=column)
df = data[["RM", "LSTAT", "MEDV"]]

df = standardize(df)
# df = minmax_normalize(df)

rm = np.array(df["RM"])
lstat = np.array(df["LSTAT"])

X = np.stack([rm, lstat], axis=-1)
y = np.array(df["MEDV"])

linear_model = MultipleLinearRegression()
linear_model.fit(X, y)


print(linear_model.coef_)
print(linear_model.intercept_)
