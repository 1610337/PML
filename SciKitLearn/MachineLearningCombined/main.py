# General Imports
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# KNN
from sklearn.neighbors import KNeighborsClassifier

# KNN Bagging
from sklearn.ensemble import BaggingClassifier

# Random Forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# SVM
from sklearn.svm import SVC

import parameters as pa

import sys


def main():

    X_train, X_test, y_train, y_test = get_filtered_data(pa.inputseparator, pa.inputfilename, pa.labelcolumn, pa.columnfilter, pa.featurecols, pa.linefilter, pa.col_names)

    #sys.exit(0)

    knn_value = knn(X_train, X_test, y_train, y_test)
    #knn_bagging_value = knn_bagging(X_train, y_train)

    sg_decision_tree_value, random_forest_value = random_forest(X_train, X_test, y_train, y_test)

    svm_value = svm(X_train, X_test, y_train, y_test)

    print("Precision weighted by avg")
    print('KNN             : ', knn_value)
    print('1 Decision Tree : ', sg_decision_tree_value)
    print('Random Forest   : ', random_forest_value)
    print('SVM             : ', svm_value)

    #print("Knn-Bagging:", knn_bagging_value)


def knn_bagging(X_train, y_train):
    np.array(X_train).reshape(-1,1)
    np.array(y_train).reshape(-1, 1)
    X_train = np.ravel(X_train)
    y_train = np.ravel(y_train)

    m = KNeighborsClassifier(n_neighbors=1)
    bag = BaggingClassifier(m, max_samples=5, max_features=2,n_jobs=2, oob_score=True)
    bag.fit(X_train, y_train)

    return bag.oob_score


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


def get_filtered_data(inputseparator, inputfilename, labelcolumn, columnfilter, featurecols, linefilter, col_names):
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
