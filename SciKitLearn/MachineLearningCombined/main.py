# General Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# KNN
from sklearn.neighbors import KNeighborsClassifier

# Random Forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# SVM
from sklearn.svm import SVC




def main():
    X_train, X_test, y_train, y_test = get_filtered_data()

    #knn(X_train, X_test, y_train, y_test)

    #random_forest(X_train, X_test, y_train, y_test)

    svm(X_train, X_test, y_train, y_test)

def knn(X_train, X_test, y_train, y_test):
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


def random_forest(X_train, X_test, y_train, y_test):
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


def svm(X_train, X_test, y_train, y_test):
    model = SVC()

    # Model get´s trained with all the default parameters and without data preparation
    # Grid-Search could be used to find better parameters
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(confusion_matrix(y_test, predictions))
    print()
    print(classification_report(y_test, predictions))

    print(type(classification_report(y_test, predictions)))
def get_filtered_data():
    inputseparator = ','  # separator for csv columns
    inputfilename = 'iris.data'  # filename input data
    labelcolumn = 4  # column of label
    columnfilter = [0, 1, 2, 3]  # columns with features
    featurecols = []  # selected Features
    linefilter = '[1,2]'  # linefilter: lines to ignore

    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    # Load Data
    filedata = [line.split(inputseparator) for line in open(inputfilename).read().split("\n") if line != '']

    # Drop rows of data
    rawdata = [[filedata[j][i] for i in range(len(filedata[j])) if i in columnfilter or i == labelcolumn] for j in
               range(len(filedata)) if eval('j not in ' + linefilter)]

    '''
        dt = {}  # Working Data; Select features
        for i in range(len(rawdata)):
        dt[i] = {'features': [rawdata[i][k] for k in featurecols], 'label': rawdata[i][labelcolumn]}
    '''
    # Create dataframe from the rawdata
    df = pd.DataFrame.from_records(rawdata, columns=col_names)
    # drop feature columns
    df = df.drop(featurecols, axis=1)

    # get a dataframe without the species column
    df_feat = pd.DataFrame(df, columns=df.columns[:-1])



    # split dataset into training and testing sets
    X = df_feat
    y = df['species']
    # random_state = seed for random values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    return X_train, X_test, y_train, y_test

main()
