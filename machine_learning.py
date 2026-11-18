
# Machine Learning Lab Programs
# Name: M. Hema Sai
# Subject Code: ITA0601

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

# SESSION 5: KNN
print("\nSESSION 5: K-Nearest Neighbours")

X = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
y = [0, 0, 0, 1, 1]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

print("Predicted Class:", knn.predict([[4, 5]])[0])


# SESSION 7: LOGISTIC REGRESSION
print("\nSESSION 7: Logistic Regression")

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

logistic = LogisticRegression()
logistic.fit(X, y)

print("Predicted Class:", logistic.predict([[5]])[0])


# SESSION 8: LINEAR REGRESSION
print("\nSESSION 8: Linear Regression")

X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

linear = LinearRegression()
linear.fit(X, y)

print("Predicted Value:", linear.predict([[6]])[0])


# SESSION 9: LINEAR VS POLYNOMIAL REGRESSION
print("\nSESSION 9: Linear vs Polynomial Regression")

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)

print("Linear Prediction:",
      linear_model.predict([[6]])[0])

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

print("Polynomial Prediction:",
      poly_model.predict(poly.transform([[6]]))[0])

# Graph
plt.scatter(X, y, label="Actual Data")
plt.plot(X, linear_model.predict(X), label="Linear")
plt.plot(X, poly_model.predict(X_poly), label="Polynomial")

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()

# MACHINE LEARNING LAB PROGRAMS
# Name: M. Hema Sai
# Subject Code: ITA0601

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# SESSION 10: EXPECTATION AND MAXIMIZATION ALGORITHM

print("\nSESSION 10: EXPECTATION AND MAXIMIZATION")

X = np.array([
    [1, 2], [2, 3], [3, 4],
    [8, 9], [9, 10], [10, 11]
])

model = GaussianMixture(n_components=2, random_state=42)
model.fit(X)

prediction = model.predict(X)

print("Cluster Labels:", prediction)
print("Means:", model.means_)
print("Model Converged:", model.converged_)


# SESSION 11: CREDIT SCORE CLASSIFICATION

print("\nSESSION 11: CREDIT SCORE CLASSIFICATION")

X = [
    [25, 30000],
    [35, 50000],
    [45, 60000],
    [22, 20000],
    [40, 70000],
    [30, 40000]
]

y = ["Good", "Good", "Good", "Poor", "Good", "Poor"]

credit_model = LogisticRegression()
credit_model.fit(X, y)

prediction = credit_model.predict([[28, 35000]])

print("Age: 28")
print("Salary: 35000")
print("Credit Score:", prediction[0])


# SESSION 12: IRIS FLOWER CLASSIFICATION USING KNN

print("\nSESSION 12: IRIS FLOWER CLASSIFICATION")

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy * 100, "%")

sample = [[5.1, 3.5, 1.4, 0.2]]
result = knn.predict(sample)

print("Predicted Flower:", iris.target_names[result[0]])

print("\nALL THREE PROGRAMS COMPLETED SUCCESSFULLY!")

