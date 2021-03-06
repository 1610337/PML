# General Imports
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# KNN
from sklearn.neighbors import KNeighborsClassifier

# Random Forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# SVM
from sklearn.svm import SVC

import DataAnalysis

import sys


def main():
    X_train, X_test, y_train, y_test, returnStr = get_filtered_data()

    knn_value = knn(X_train, X_test, y_train, y_test)

    sg_decision_tree_value, random_forest_value = random_forest(X_train, X_test, y_train, y_test)

    svm_value = svm(X_train, X_test, y_train, y_test)

    bagging_value = ensemble_learning(X_train, X_test, y_train, y_test)


    of = open("iris_outputfile.txt", 'a') # a to write inside
    print("*"*60, file=of)
    print("", file=of)

    of.write(returnStr)
    print('Particular Data Analysis (with kNN) [1.2]\n', file=of)

    print("Precision weighted by avg")
    print('KNN             : ', knn_value)
    # print('1 Decision Tree : ', sg_decision_tree_value)
    print('Random Forest   : ', random_forest_value)
    print('SVM             : ', svm_value)
    print("Bagging         : ", bagging_value)

    print("Precision weighted by avg", file=of)
    print('KNN             : ', knn_value, file=of)
    print('Random Forest   : ', random_forest_value, file=of)
    print('SVM             : ', svm_value, file=of)
    print("Bagging         : ", bagging_value, file=of)

def ensemble_learning(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=1)
    dtree = DecisionTreeClassifier()
    supportvc = SVC(gamma='auto')

    clf1 = VotingClassifier(estimators=[('knn', knn), ('dtree', dtree), ('supportvc', supportvc)], voting='hard')
    clf1.fit(X_train, y_train)

    predictions = clf1.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True)

    return report['weighted avg']['precision']


def knn(X_train, X_test, y_train, y_test):
    # knn model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    #pred = knn.predict(X_test)

    #  the number of predicted classes which ended up in a wrong classification bin based on the true classes
    # print(confusion_matrix(y_test, pred))
    # Build a text report showing the main classification metrics (like precision)
    #report = classification_report(y_test, pred, output_dict=True)

    ## See how the model performs on the test data.
    knn_score = knn.score(X_test, y_test)

    #print(knn_score)

    #print(report)

    return knn_score


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
    model = SVC(gamma='auto')

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

    df, returnStr = DataAnalysis.get_filtered_data()

    print(df.head())
    returnStr += str(df.head())


    # create a new feature
    print(df.head())
    df['CalcFeature'] = df['petal_length']*df['petal_width']
    df = df.drop(['petal_length', 'petal_width'], axis=1)
    print(df.head())

    # get a dataframe without the species column
    df_feat = df.drop(columns=['species'])

    # split dataset into training and testing sets
    X = df_feat

    y = df['species']
    # random_state = seed for random values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test, returnStr

main()
