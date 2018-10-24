import csv
import random
import math
import operator
import numpy as np
import pandas as pd

import sys


def testMain():
    d = pd.Series([1,2,3,4],['a', 'b', 'c', 'd'])
    print(d)

    df = pd.DataFrame(2, ['a', 'b'], ['a1', 'b1'])
    print(df)
    pass

def main() :
    print("Implementation of the KNN Algorithm für the IRIS-Dataset")
    # TODO check for results ... there are values with 100% accuracy
    # TODO Change output to pandas DF

    mainDataFrame = []
    for k in range(3, 10):
        tempListForKResults = []
        tempSeries = pd.Series()
        for split in np.arange(0.1, 0.99, 0.01):
            # print(round(split,2))
            # now we´re executing the following code for each k - split combination
            # which is essentially the same code as in the previous example

            # prepare data
            trainingSet = []
            testSet = []
            split = round(split, 2)
            loadDataset('iris.csv', split, trainingSet, testSet)
            print('Train set: ' + repr(len(trainingSet)))
            print('Test set: ' + repr(len(testSet)))
            # generate predictions
            predictions = []

            for x in range(len(testSet)):
                neighbors = getNeighbors(trainingSet, testSet[x], k)
                result = getResponse(neighbors)
                predictions.append(result)
                # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
            accuracy = getAccuracy(testSet, predictions)
            print('Accuracy: ' + repr(accuracy) + '%' + ' K value -->' + str(k) + ' Split Value -->' + str(round(split,2)))

            tempListForKResults.append(repr(accuracy))
            #tempSeries += pd.Series([repr(accuracy)])
            tempSeries = tempSeries.append(pd.Series([repr(accuracy)], [round(split,2)]))

        print(tempSeries)
        # print(tempListForKResults)
        mainDataFrame.append(tempListForKResults)
        # sys.exit(0)
    printOutMainDF(mainDataFrame)

def printOutMainDF(df):
    print('*' * 45)
    # print(df)
    for set in df:
        for acc in set:
            print(acc[:3], end="")
        print()

    print('*'*45)

def testAllFunctions():
    trainingSet = []
    testSet = []
    loadDataset('iris.csv', 0.66, trainingSet, testSet)

    print('Train: ' + repr(len(trainingSet)))
    print('Test: ' + repr(len(testSet)))

    # testing of the euclideanDistance function
    data1 = [2, 2, 2, 'a']
    data2 = [4, 4, 4, 'b']
    distance = euclideanDistance(data1, data2, 3)
    print('Distance: ' + repr(distance))

    # testing of the distance function
    trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
    testInstance = [5, 5, 5]
    k = 1
    neighbors = getNeighbors(trainSet, testInstance, 1)
    print(neighbors)

    # testing get Response function
    neighbors = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
    response = getResponse(neighbors)
    print(response)

    # test accuracy method
    testSet = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
    predictions = ['a', 'a', 'a']
    accuracy = getAccuracy(testSet, predictions)
    print(accuracy)

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    # splits iris dateset randomly into train and test datasets using provided split ratio.

    # added while loop which only stops if the test data set is not 0
    exitCond = True
    while exitCond:
        with open(filename, 'r', encoding='utf-8-sig') as csvfile:
            lines = csv.reader(csvfile)
            dataset = list(lines)
            for x in range(len(dataset)-1):
                for y in range(4):
                    # save each value as a float and not string
                    dataset[x][y] = float(dataset[x][y])
                if random.random() < split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])
        if len(testSet)!= 0:
            exitCond = False

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


main()
testMain()