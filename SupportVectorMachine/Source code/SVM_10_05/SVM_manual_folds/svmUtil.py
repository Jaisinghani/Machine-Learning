from sklearn.preprocessing import StandardScaler
from numpy import *
from sklearn.metrics import classification_report
from sklearn.model_selection import *
import time
from sklearn import metrics


def getfeatures_and_types(glassdataset):
    glassFeatures = glassdataset[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].values

    glassType = glassdataset['Type'].values
    return glassFeatures,glassType

def normalizedata(dataset):
    ss = StandardScaler()
    ss.fit(dataset)
    normalized_dataset = ss.transform(dataset)
    return normalized_dataset



