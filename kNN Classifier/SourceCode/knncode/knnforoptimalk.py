from numpy import *
import numpy as np
import knncode.kNNClassifier as knn

optimal_k = 4
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
np.random.shuffle(copyOfTrainingDataSet)

#running knn for optimal K
predictor = knn.MNISTPredictor(copyOfTrainingDataSet, optimal_k)
optimal_k_predictions = knn.predict_test_set(predictor, testImagesSet)
print("Predictions for Optimal k (4) is : ",optimal_k_predictions )


#Accuracy and error calculations
optimal_k_accuracy = knn.evaluate_prediction(optimal_k_predictions, testLabelsSet)
optimal_k_classification_accuracy  = optimal_k_accuracy*100
print(" Classification accuracy for optimal K (4) :", optimal_k_classification_accuracy)
optimal_k_classification_error = (1-optimal_k_accuracy)*100
print(" Classification error for optimal K (4) :", optimal_k_classification_error)

#confidence interval for optimal k
ci = knn.confidenceinterval(optimal_k_classification_accuracy,testImagesSet)
print("Confidence Interval for optimal k (4) : ",ci)



#Confusion matrix for optimal k predictions
cm = knn.confusionmatrix(optimal_k_predictions,testLabelsSet)
print("confusion matrix for optimal-K (4) : ",cm)
