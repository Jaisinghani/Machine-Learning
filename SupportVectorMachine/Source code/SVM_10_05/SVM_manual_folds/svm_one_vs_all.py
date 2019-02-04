from sklearn.model_selection import StratifiedKFold
import datareader as datareader
import svmUtil as svmUtil
from sklearn.utils import shuffle
from numpy import *
import time
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


#DataHandling
glass_dataset_filepath = './UCIGlassDataset/glass.csv'
glass_dataset = datareader.readGlassDataSet(glass_dataset_filepath)

#Shuffling the dataset
shuffled_dataset = shuffle(glass_dataset, random_state = 1)

#separating the types and features
(glass_dataset_features, glass_type) = svmUtil.getfeatures_and_types(glass_dataset)


#Normalizing the features
normalized_glass_features = svmUtil.normalizedata(glass_dataset_features)

#HyperParameters
cvalues = []
gammavalues=[]
degreevalues = []
coefvalues = []

for x in range(-5,16):
    cvalues.append(2**x)
print("cvalues :", cvalues)

for y in range(-15,4):
    gammavalues.append(2**y)
print("gammavalues :", gammavalues)

for z in range (2, 6):
    degreevalues.append(z)
print("degreevalues :", degreevalues)

for u in range (-5, 1):
    coefvalues.append(2**u)
print("coefvalues :", coefvalues)

def linearKernel(parameters):
    test_accuracy = []
    test_accuracy_with_params = []
    k_test_fold = StratifiedKFold(5)
    for (train, test) in (k_test_fold.split(normalized_glass_features, glass_type)):


        #test fold
        test_dataset_features = normalized_glass_features[test]
        test_dataset_types = glass_type[test]

        #Rest of the data
        training_dataset_features = normalized_glass_features[train]
        training_dataset_types = glass_type[train]

        #Splitting rest of the data into 80-20% such as 20% for validation set
        (training_features,
         validation_features,
         training_glassType,
         validation_glassType) = train_test_split(training_dataset_features, training_dataset_types, train_size=0.80, random_state=1)

        validation_acuracies = []
        start_time_training = time.clock()

        #Training different models with different hyperparameters
        for cvalue in cvalues:
            classifier = OneVsRestClassifier(SVC(kernel="linear", C=cvalue))
            classifier.fit(training_features,training_glassType)
            validation_true, validation_pred = validation_glassType, classifier.predict(validation_features)
            accuracy_Validationset = metrics.accuracy_score(validation_true, validation_pred)
            validation_acuracies.append((classifier.get_params().get('estimator__C'),accuracy_Validationset))
        validation_acuracies.sort(key=lambda val: val[1])
        print("sorted validation_acuracies :", validation_acuracies)

        #optimal hyperparameter with accuracy
        print(" optimal hyperparameter with accuracy is :", validation_acuracies[-1])

        #Training a new model on the entire 4 folds with optimal Hyper Parameters
        classifier_1 = OneVsRestClassifier(SVC(kernel="linear", C=validation_acuracies[-1][0]))
        classifier_1.fit(training_dataset_features,training_dataset_types)

        end_time_training = time.clock()
        print("Time taken to train for one fold for linear kernel is :", (end_time_training- start_time_training))

        test_true, test_pred = test_dataset_types, classifier_1.predict(test_dataset_features)
        accuracy_test = metrics.accuracy_score(test_true, test_pred)
        print("accuracy_test  :", accuracy_test)
        test_accuracy_with_params.append((classifier_1.get_params().get('estimator__C'), accuracy_test))
        test_accuracy.append(accuracy_test)
    print("Test Accuracies for all fold with params :", test_accuracy_with_params)
    print("Test Accuracies for all fold:", test_accuracy)
    average_accuracy = sum(test_accuracy) / len(test_accuracy)
    print("average accuracy for linear SVM is :", average_accuracy)



def rbfKernel(parameters):
    test_accuracy = []
    test_accuracy_with_params = []
    k_test_fold = StratifiedKFold(5)
    for (train, test) in (k_test_fold.split(normalized_glass_features, glass_type)):


        #test fold
        test_dataset_features = normalized_glass_features[test]
        test_dataset_types = glass_type[test]

        #Rest of the data
        training_dataset_features = normalized_glass_features[train]
        training_dataset_types = glass_type[train]

        #Splitting rest of the data into 80-20% such as 20% for validation set
        (training_features,
         validation_features,
         training_glassType,
         validation_glassType) = train_test_split(training_dataset_features, training_dataset_types, train_size=0.80, random_state=1)

        validation_acuracies = []
        start_time_training = time.clock()

        #Training different models with different hyperparameters
        for cvalue in cvalues:
            for gammavalue in gammavalues:
                classifier = OneVsRestClassifier(SVC(kernel="rbf", C=cvalue, gamma =gammavalue))
                classifier.fit(training_features,training_glassType)
                validation_true, validation_pred = validation_glassType, classifier.predict(validation_features)
                accuracy_Validationset = metrics.accuracy_score(validation_true, validation_pred)
                validation_acuracies.append((classifier.get_params().get('estimator__C'), classifier.get_params().get('estimator__gamma'),accuracy_Validationset))
        validation_acuracies.sort(key=lambda val: val[2])
        print("sorted validation_acuracies :", validation_acuracies)

        #optimal hyperparameter with accuracy
        print(" optimal hyperparameter with accuracy is :", validation_acuracies[-1])

        #Training a new model on the entire 4 folds with optimal Hyper Parameters
        classifier_1 = OneVsRestClassifier(SVC(kernel="rbf", C=validation_acuracies[-1][0], gamma=validation_acuracies[-1][1]))
        classifier_1.fit(training_dataset_features,training_dataset_types)

        end_time_training = time.clock()
        print("Time taken to train for one fold for RBF kernel is :", (end_time_training- start_time_training))

        test_true, test_pred = test_dataset_types, classifier_1.predict(test_dataset_features)
        accuracy_test = metrics.accuracy_score(test_true, test_pred)
        print("accuracy_test  :", accuracy_test)
        test_accuracy_with_params.append((classifier_1.get_params().get('estimator__C'), classifier_1.get_params().get('estimator__gamma'),accuracy_test))
        test_accuracy.append(accuracy_test)
    print("Test Accuracies for all fold with params :", test_accuracy_with_params)
    print("Test Accuracies for all fold:", test_accuracy)
    average_accuracy = sum(test_accuracy) / len(test_accuracy)
    print("average accuracy for RBF SVM is :", average_accuracy)

def polynomialKernel(parameters):
    test_accuracy = []
    test_accuracy_with_params = []
    k_test_fold = StratifiedKFold(5)
    for (train, test) in (k_test_fold.split(normalized_glass_features, glass_type)):


        #test fold
        test_dataset_features = normalized_glass_features[test]
        test_dataset_types = glass_type[test]

        #Rest of the data
        training_dataset_features = normalized_glass_features[train]
        training_dataset_types = glass_type[train]

        #Splitting rest of the data into 80-20% such as 20% for validation set
        (training_features,
         validation_features,
         training_glassType,
         validation_glassType) = train_test_split(training_dataset_features, training_dataset_types, train_size=0.80, random_state=1)

        validation_acuracies = []
        start_time_training = time.clock()

        #Training different models with different hyperparameters
        for cvalue in cvalues:
            for gammavalue in gammavalues:
                for degreevalue in degreevalues:
                    for coefvalue in coefvalues:
                        classifier = OneVsRestClassifier(SVC(kernel="poly", C=cvalue, gamma =gammavalue, degree=degreevalue, coef0= coefvalue))
                        classifier.fit(training_features,training_glassType)
                        validation_true, validation_pred = validation_glassType, classifier.predict(validation_features)
                        accuracy_Validationset = metrics.accuracy_score(validation_true, validation_pred)
                        validation_acuracies.append((classifier.get_params().get('estimator__C'), classifier.get_params().get('estimator__gamma'),classifier.get_params().get('estimator__degree'), classifier.get_params().get('estimator__coef0'), accuracy_Validationset))
        validation_acuracies.sort(key=lambda val: val[4])
        print("sorted validation_acuracies :", validation_acuracies)

        #optimal hyperparameter with accuracy
        print(" optimal hyperparameter with accuracy is :", validation_acuracies[-1])

        #Training a new model on the entire 4 folds with optimal Hyper Parameters
        classifier_1 = OneVsRestClassifier(SVC(kernel="poly", C=validation_acuracies[-1][0], gamma=validation_acuracies[-1][1], degree=validation_acuracies[-1][2], coef0=validation_acuracies[-1][3]))
        classifier_1.fit(training_dataset_features,training_dataset_types)

        end_time_training = time.clock()
        print("Time taken to train for one fold for polynomial kernel is :", (end_time_training- start_time_training))

        test_true, test_pred = test_dataset_types, classifier_1.predict(test_dataset_features)
        accuracy_test = metrics.accuracy_score(test_true, test_pred)
        print("accuracy_test  :", accuracy_test)
        test_accuracy_with_params.append((classifier_1.get_params().get('estimator__C'), classifier_1.get_params().get('estimator__gamma'),classifier_1.get_params().get('estimator__degree'),classifier_1.get_params().get('estimator__coef0'),accuracy_test))
        test_accuracy.append(accuracy_test)
    print("Test Accuracies for all fold with params :", test_accuracy_with_params)
    print("Test Accuracies for all fold:", test_accuracy)
    average_accuracy = sum(test_accuracy) / len(test_accuracy)
    print("average accuracy for polynomial SVM is :", average_accuracy)



def sigmoidKernel(parameters):
    test_accuracy = []
    test_accuracy_with_params = []
    k_test_fold = StratifiedKFold(5)
    for (train, test) in (k_test_fold.split(normalized_glass_features, glass_type)):


        #test fold
        test_dataset_features = normalized_glass_features[test]
        test_dataset_types = glass_type[test]

        #Rest of the data
        training_dataset_features = normalized_glass_features[train]
        training_dataset_types = glass_type[train]

        #Splitting rest of the data into 80-20% such as 20% for validation set
        (training_features,
         validation_features,
         training_glassType,
         validation_glassType) = train_test_split(training_dataset_features, training_dataset_types, train_size=0.80, random_state=1)

        validation_acuracies = []
        start_time_training = time.clock()

        #Training different models with different hyperparameters
        for cvalue in cvalues:
            for gammavalue in gammavalues:
                for coefvalue in coefvalues:
                    classifier = OneVsRestClassifier(SVC(kernel="sigmoid", C=cvalue, gamma =gammavalue, coef0= coefvalue))
                    classifier.fit(training_features,training_glassType)
                    validation_true, validation_pred = validation_glassType, classifier.predict(validation_features)
                    accuracy_Validationset = metrics.accuracy_score(validation_true, validation_pred)
                    validation_acuracies.append((classifier.get_params().get('estimator__C'), classifier.get_params().get('estimator__gamma'), classifier.get_params().get('estimator__coef0'), accuracy_Validationset))
        validation_acuracies.sort(key=lambda val: val[3])
        print("sorted validation_acuracies :", validation_acuracies)

        #optimal hyperparameter with accuracy
        print(" optimal hyperparameter with accuracy is :", validation_acuracies[-1])

        #Training a new model on the entire 4 folds with optimal Hyper Parameters
        classifier_1 = OneVsRestClassifier(SVC(kernel="sigmoid", C=validation_acuracies[-1][0], gamma=validation_acuracies[-1][1], coef0=validation_acuracies[-1][2]))
        classifier_1.fit(training_dataset_features,training_dataset_types)

        end_time_training = time.clock()
        print("Time taken to train for one fold for sigmoid kernel is :", (end_time_training- start_time_training))

        test_true, test_pred = test_dataset_types, classifier_1.predict(test_dataset_features)
        accuracy_test = metrics.accuracy_score(test_true, test_pred)
        print("accuracy_test  :", accuracy_test)
        test_accuracy_with_params.append((classifier_1.get_params().get('estimator__C'), classifier_1.get_params().get('estimator__gamma'),classifier_1.get_params().get('estimator__coef0'), accuracy_test))
        test_accuracy.append(accuracy_test)
    print("Test Accuracies for all fold with params :", test_accuracy_with_params)
    print("Test Accuracies for all fold:", test_accuracy)
    average_accuracy = sum(test_accuracy) / len(test_accuracy)
    print("average accuracy for sigmoid SVM is :", average_accuracy)

#linear
parameters_linear = [{'C': cvalues}]
linearKernel(parameters_linear)

#RBF
parameters_RBF = [{ 'gamma': gammavalues, 'C': cvalues}]
rbfKernel(parameters_RBF)

#Polynomial
parameters_poly = [{ 'gamma': gammavalues, 'C': cvalues, 'degree' : degreevalues, 'coef0': coefvalues}]
polynomialKernel(parameters_poly)

#Sigmoid
parameters_sigmoid = [{ 'gamma': gammavalues, 'C': cvalues, 'coef0': coefvalues}]
sigmoidKernel(parameters_sigmoid)

