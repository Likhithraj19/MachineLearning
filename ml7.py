import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Importing iris dataset from sklearn and splitting input and output
iris = load_iris()
X = iris.data
y = iris.target

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and training the KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = knn_model.predict(X_test)

# Calculating and printing the accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
