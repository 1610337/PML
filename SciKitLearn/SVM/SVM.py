import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

iris = datasets.load_iris()

print(iris['DESCR'])

df_feat = pd.DataFrame(iris['data'], columns=iris['feature_names'])

print(iris['target'])


X = df_feat
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = SVC()

# Model get´s trained with all the default parameters and without data preparation
# Grid-Search could be used to find better parameters
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print()
print(classification_report(y_test, predictions))