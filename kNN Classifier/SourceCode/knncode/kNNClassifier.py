from numpy import *
import numpy as np
from collections import defaultdict

#kNNClassifier
class Knnclassifier:
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k

    def predict(self, point):
        distances = []
        for index, value in enumerate(self.dataset):
            distance = self.distance(value[0], point)
            distances.append((distance, index))

        distances.sort(key=lambda val: val[0])

        indexesOfKImages = []
        for i in range(self.k):
            indexesOfKImages.append(distances[i][1])

        labels = []
        for j in indexesOfKImages:
            labels.append(self.dataset[j][1])

        prediction = self.observe(labels)
        return prediction

    def predictionslidingwindow(self, point):
        distances = []
        for index, value in enumerate(self.dataset):
            train_image = value[0]

            # padded training image to convert from 28*28 to 30*30
            paddedImage = np.pad(train_image, [(1, 1), (1, 1)], mode='constant')

            # slice train image into 9 images to find min distance
            sliding_window_distances = []
            for i in range(3):
                for j in range(3):
                    slicedImage = paddedImage[i:i + 28, j:j + 28]
                    distance = self.distance(slicedImage, point)
                    sliding_window_distances.append(distance)
            distances.append((min(sliding_window_distances), index))
        distances.sort(key=lambda val: val[0])

        indexesOfKImages = []
        for i in range(self.k):
            indexesOfKImages.append(distances[i][1])

        labels = []
        for j in indexesOfKImages:
            labels.append(self.dataset[j][1])

        prediction = self.observe(labels)
        return prediction


#Predictore class used to predict test images
class MNISTPredictor(Knnclassifier):

    def distance(self, p1, p2):
        dist = calEuclideanDistance(p1, p2)
        return dist

    def observe(self, values):
        return get_majority(values)

#function to calculate euclidean distance
def calEuclideanDistance(point1, point2):
    distance = np.sqrt(np.sum((point1 - point2) ** 2))
    return distance

#function to count majority votes to select a label based on the predictions
def get_majority(votes):
    counter = defaultdict(int)
    for vote in votes:
        counter[vote] += 1

    majority_count = max(counter.values())
    for key, value in counter.items():
        if value == majority_count:
            return key


#function used to predict labels for test images
def predict_test_set(predictor, test_set):
    """Compute the prediction for every element of the test set."""
    predictions = [predictor.predict(test_set[i]) for i in range(len(test_set))]
    return predictions

#function used to evaluate prediction and calculate accuracy
def evaluate_prediction(predictions, answers):
    correct = sum(asarray(predictions) == asarray(answers))
    total = float(prod(len(answers)))
    return correct / total


#calculates the confidence interval
def confidenceinterval(classificationaccuracy,testImageSet ):
    kaccuracy = classificationaccuracy/100
    confidence_interval = []
    confidence_interval_pos = kaccuracy+1.96* (sqrt((kaccuracy*(1-kaccuracy))/len(testImageSet)))
    confidence_interval.append(confidence_interval_pos)
    confidence_interval_neg = kaccuracy-1.96* (sqrt((kaccuracy*(1-kaccuracy))/len(testImageSet)))
    confidence_interval.append(confidence_interval_neg)
    return confidence_interval



#Confusion matrix calculation
def confusionmatrix(predictions, actuallabels):
    row = {}
    for i in range(len(actuallabels)):
        x = str(actuallabels[i]) + str(predictions[i])
        key = "group_{0}".format(x)
        if key in row:
            row["group_{0}".format(x)] = row["group_{0}".format(x)] + 1
        else:
            row["group_{0}".format(x)] = 1

    labelrows = []
    for x in range(0, 10):
        for y in range(0, 10):
            j = str(x) + str(y)
            p = "group_{0}".format(j)
            if p in row:
                labelrows.append(row["group_{0}".format(j)])
            else:
                labelrows.append(0)

    cm = reshape(labelrows, (10, 10))
    return cm
