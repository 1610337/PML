import csv
import random
import math
import operator


# The following implementation was according to the following tutorial
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# Only a few changes for python 3.7 were made

def main() :
    print("Implementation of the KNN Algorithm for the IRIS-Dataset")

    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('iris.csv', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))

    # generate predictions
    predictions = []
    k = 3
    #for k in range(3,10):
    #    for split in range(0.01, 0.99, 0.001):
    #        print(split)

    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

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