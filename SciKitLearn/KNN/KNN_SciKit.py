import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# reading and setting up the data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, header=None, names=col_names)
# print(df.head())

print(df.head())

# get a dataframe without the species column
df_feat = pd.DataFrame(df, columns=df.columns[:-1])
# print(df_feat.head())

# split dataset into training and testing sets
X = df_feat
y = df['species']
# random_state = seed for random values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# knn model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

#  the number of predicted classes which ended up in a wrong classification bin based on the true classes
print(confusion_matrix(y_test, pred))
# Build a text report showing the main classification metrics (like precision)
print(classification_report(y_test, pred))

# using "elbow-method" to choose correct K-value (even tho we have
# the highest prediction rate already
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    # this is basically the average error rate
    # where the prediction wasn't exactly equal to
    # the test values
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue',
         linestyle='dashed', marker='o', markerfacecolor='red',
         markersize=10)
plt.title("Error Rate vs. K Value")
plt.xlabel('k')
plt.ylabel('error rate')
plt.savefig('foo.pdf')
