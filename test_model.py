import os
import pickle

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MaxAbsScaler

from featureExtraction import (extractFeatures, extractFeaturesSilence,
                               slidingObsWindow)


def predict(folder):

    global model
    global trainScaler
    global trainPCA

    totalAnom = 0
    totalSessions = 0

    for file in os.listdir(folder):
        print(file)
        test = np.loadtxt(open(folder+"/"+file, 'r'), dtype=int)
        windows = np.array(slidingObsWindow(test, 3, 1))
        features_timedependent = extractFeatures(windows)
        features_timeindependent = extractFeaturesSilence(windows)
        test_features = np.hstack((features_timedependent, features_timeindependent))

        test_featuresN=trainScaler.transform(test_features)
        test_featuresNPCA = trainPCA.transform(test_featuresN)

        a = model.predict(test_featuresNPCA)
        b = model.decision_function(test_featuresNPCA)

        totalSessions += 1
        anom = 0
        if len(test)-3>15:
            for i,prediction in enumerate(a):
                if prediction == -1 and b[i] <= 0:
                    anom += 1
            
            if anom/len(a) >0.5:
                totalAnom += 1
                print(a)
                print(b)
                print("ANOMALY")
            else:
                print(np.average(b))
                print("OK")

    return (totalSessions,totalAnom)

def main():

    global model
    global trainScaler
    global trainPCA

    with open('scaler', 'rb') as pickle_file:
        trainScaler = pickle.load(pickle_file)
    with open('pca', 'rb') as pickle_file:
        trainPCA = pickle.load(pickle_file)
    with open('model', 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    #testing ok/non anomaly sessions
    print("Predicting on ok sessions")
    p = predict("streams_ok")
    
    total = p[0]
    anomalies = p[1]
    ok = total - anomalies
    FP = anomalies
    TN = ok
    print("True negatives: "+str(TN))
    print("False positives: "+str(FP))
    print("Percentage of false positives: "+str(FP/total))
    

if __name__ == '__main__':
    main()
