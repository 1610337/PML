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
    # print(confusion_matrix(y_test, pred))
    # Build a text report showing the main classification metrics (like precision)
    report = classification_report(y_test, pred, output_dict=True)

    return report['weighted avg']['precision']


def random_forest(X_train, X_test, y_train, y_test):
    # training a single decision tree:
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)

    predictions = dtree.predict(X_test)
    # print("Single Decision Tree:")
    # print(confusion_matrix(y_test, predictions))
    # print('\n')
    # print(classification_report(y_test, predictions))
    report1 = classification_report(y_test, predictions, output_dict=True)

    # now training an actual random forest model
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)

    rfc_pred = rfc.predict(X_test)
    # print("Random Forest:")
    # print(confusion_matrix(y_test, rfc_pred))
    # print('\n')
    # print(classification_report(y_test, rfc_pred))
    report2 = classification_report(y_test, rfc_pred, output_dict=True)

    return report1['weighted avg']['precision'], report2['weighted avg']['precision']

def svm(X_train, X_test, y_train, y_test):
    model = SVC()

    # Model get´s trained with all the default parameters and without data preparation
    # Grid-Search could be used to find better parameters
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # print(confusion_matrix(y_test, predictions))
    # print()
    # print(classification_report(y_test, predictions))
    report = classification_report(y_test, predictions, output_dict=True)
    return report['weighted avg']['precision']


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
