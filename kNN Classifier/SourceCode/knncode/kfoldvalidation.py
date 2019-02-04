from numpy import *
import numpy as np
import itertools
import knncode.kNNClassifier as knn

#Get the training images and labels
import knncode.datareader as datareader
print("loading training images")
trainImagesSet = datareader.parse_images("train-images-idx3-ubyte")
print("size of train image set---->", size(trainImagesSet))
print("shape of train image set---->", shape(trainImagesSet))

print("loading training labels")
trainLabelsSet = datareader.parse_labels("train-labels-idx1-ubyte")
print("size of train image set---->", size(trainLabelsSet))
print("shape of train image set---->", shape(trainLabelsSet))


# Shuffling the training Image Set
dataset = []
for i in range(len(trainImagesSet)):
    dataset.append((trainImagesSet[i, :, :], trainLabelsSet[i]))

#Copy of training image set
copyOfTrainingDataSet = dataset[:]
np.random.shuffle(copyOfTrainingDataSet)

# Creating 10-folds
foldedArray = []
start = 0
end = 6000
i = 0
for i in range(10):
    arr = copyOfTrainingDataSet[start:end]
    start += 6000
    end += 6000
    foldedArray.append(arr)

# 10-CROSS VALIDATION
k_accuracy = []

ks = [1,2,3,4,5,6,7,8,9,10]
all_k_all_fold = []
for k in ks:

    # generating test data and training data from 10 folds
    fold_10_accuracy_each_k = []
    for fold in range(len(foldedArray)):
        training_set = []
        training_set = foldedArray[:]
        test_labels = []
        testing_set = []
        train_set_fold_copy = training_set[fold]
        test_set = train_set_fold_copy[:]

        for i in range(len(test_set)):
            test_labels.append(test_set[i][1])
            testing_set.append(test_set[i][0])

        del training_set[fold]
        fold_training_set = list(itertools.chain.from_iterable(training_set))
        predictor = knn.MNISTPredictor(fold_training_set, k)
        one_train_predictions = knn.predict_test_set(predictor, testing_set)
        one_train_set_accuracy = knn.evaluate_prediction(one_train_predictions, test_labels)
        fold_10_accuracy_each_k.append(one_train_set_accuracy)

    print("Accuracies for all folds for k = ",k," is :", fold_10_accuracy_each_k)
    all_k_all_fold.append((k,fold_10_accuracy_each_k))
    average_accuracy = sum(fold_10_accuracy_each_k) / 10
    print("Average_accuracy for k = ",k," is :", average_accuracy)
    accuracy_in_percent = average_accuracy * 100
    k_accuracy.append((k,accuracy_in_percent))

print("Accuracy for each fold of every k : ",all_k_all_fold)
print("Accuracies for all k ranging from 1 to 10 is :", k_accuracy)
k_accuracy.sort(key=lambda val: val[1])
print("optimal k :", k_accuracy[-1][0])