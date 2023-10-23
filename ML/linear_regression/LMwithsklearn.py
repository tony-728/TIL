import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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
data = pd.read_csv("housing.csv", names=column)

df = data[["RM", "LSTAT", "MEDV"]]

for c in ["RM", "LSTAT", "MEDV"]:
    m = df[c].mean()
    s = df[c].std()

    df[c] = (df[c] - m) / s

rm = np.array(df["RM"])
lstat = np.array(df["LSTAT"])

X = np.stack([rm, lstat], axis=-1)
y = np.array(df["MEDV"])

reg = LinearRegression().fit(X, y)
print(reg.coef_, reg.intercept_)
