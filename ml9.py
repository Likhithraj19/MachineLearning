import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
#importing iris dataset from sklearn and spliting input and output
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
Performing Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred=dtc.predict(X_test)
#Checking accuracy
acc=accuracy_score(y_test,y_pred)
print("Accuracy of model=", acc)
Output:
Accuracy: 1.0
#Visualizing decision tree
plt.figure(figsize=(12, 8))
plot_tree(dtc, feature_names=iris.feature_names,
class_names=iris.target_names, filled=True)
plt.show()
