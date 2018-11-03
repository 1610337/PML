import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# reading and setting up the data and splitting into training and test sets
# for comments on this see the implementation of KNN
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, header=None, names=col_names)
print("Data loaded")
df_feat = pd.DataFrame(df, columns=df.columns[:-1])
X = df_feat
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# training a single decision tree:
# TODO Visualize that tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)
print("Single Decision Tree:")
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

# now training an actual random forest model
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
print("Random Forest:")
print(confusion_matrix(y_test, rfc_pred))
print('\n')
print(classification_report(y_test, rfc_pred))

