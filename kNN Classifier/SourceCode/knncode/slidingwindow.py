from numpy import *
import numpy as np
import knncode.kNNClassifier as knn

optimal_k = 4
#Get the training images and labels
import knncode.datareader as datareader

def predict_test_set(predictor, test_set):
    """Compute the prediction for every element of the test set."""
    predictions = [predictor.predictionslidingwindow(test_set[i]) for i in range(len(test_set))]
    return predictions

print("loading training images")
trainImagesSet = datareader.parse_images("train-images-idx3-ubyte")
print("size of train image set---->", size(trainImagesSet))
print("shape of train image set---->", shape(trainImagesSet))

print("loading training labels")
trainLabelsSet = datareader.parse_labels("train-labels-idx1-ubyte")
print("size of train image set---->", size(trainLabelsSet))
print("shape of train image set---->", shape(trainLabelsSet))

print("loading test images")
testImagesSet = datareader.parse_images("t10k-images-idx3-ubyte")
print("size of test image set---->", size(testImagesSet))
print("shape of test image set---->", shape(testImagesSet))

print("loading test labels")
testLabelsSet = datareader.parse_labels("t10k-labels-idx1-ubyte")
print("size of test image set---->", size(testLabelsSet))
print("shape of test image set---->", shape(testLabelsSet))


# Shuffling the training Image Set
dataset = []
for i in range(len(trainImagesSet)):
    dataset.append((trainImagesSet[i, :, :], trainLabelsSet[i]))

#Copy of training image set
copyOfTrainingDataSet = dataset[:]
#np.random.shuffle(copyOfTrainingDataSet)

testdataset = []
for i in range(len(testImagesSet):
    testdataset.append((testImagesSet[i, :, :], testLabelsSet[i]))

test_labels = []
test_images = []
for i in range(len(testdataset)):
    test_labels.append(testdataset[i][1])
    test_images.append(testdataset[i][0])

#Sliding Window code
sliding_win_predictor = knn.MNISTPredictor(copyOfTrainingDataSet, optimal_k)
slinding_win_predictions = predict_test_set(sliding_win_predictor, test_images)

#Accuracy and error calculations
sliding_win_accuracy = knn.evaluate_prediction(slinding_win_predictions, test_labels)
sliding_win_classification_accuracy  = sliding_win_accuracy*100
print("Sliding window classification accuracy for optimal K (4) :", sliding_win_classification_accuracy)
sliding_win_classification_error = (1-sliding_win_accuracy)*100
print("Sliding Window classification error for optimal K (4):", sliding_win_classification_error)

#confidence interval for sliding window
sliding_win_ci = knn.confidenceinterval(sliding_win_classification_accuracy,test_images)
print("Sliding Window confidence Interval for optimal k (4): ",sliding_win_ci)

#Confusion matrix for sliding window predictions
sliding_win_cm = knn.confusionmatrix(slinding_win_predictions,test_labels)
print("confusion matrix for sliding window : ",sliding_win_cm)