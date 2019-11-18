import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data_file = pd.read_csv("student-mat.csv", delimiter=';')
data = data_file[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# print(data_file.head())
# print(data.head())

predictLabel = "G3"

x = np.array(data.drop([predictLabel], 1))
y = np.array(data[predictLabel])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print("Coefficients: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(int(round(predictions[x])), x_test[x], y_test[x])
