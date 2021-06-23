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

    features = []

    for file in os.listdir(folder):
        print(file)
        test = np.loadtxt(open(folder+"/"+file, 'r'), dtype=int)
        windows = np.array(slidingObsWindow(test, 3, 1))
        features_timedependent = extractFeatures(windows)
        features_timeindependent = extractFeaturesSilence(windows)
        test_features = np.hstack((features_timedependent, features_timeindependent))

        test_featuresN=trainScaler.transform(test_features)
        test_featuresNPCA = trainPCA.transform(test_featuresN)

        features.append(test_featuresNPCA)
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
    
    total_ok = p[0]
    anomalies = p[1]
    ok = total_ok - anomalies
    FP = anomalies
    TN = ok
    print("True negatives: "+str(TN))
    print("False positives: "+str(FP))
    print("Percentage of false positives: "+str(FP/total_ok))

    #testing anomalous sessions
    
    print("Predicting on anomalous sessions")
    p = predict("streams_anom")
    total_anom = p[0]
    anomalies = p[1]
    ok = total_anom - anomalies
    TP = anomalies
    FN = ok
    print("True positive: "+str(TP))
    print("False negative: "+str(FN))
    print("Percentage of false negatives: "+str(FN/total_ok))

    accuracy= (TP+TN)/(TP+TN+FP+FN)
    precision= TP/(TP+FP)
    recall= TP/(TP+FN)
    f1_Score = 2*(recall * precision) / (recall + precision)

    print("Accuracy: "+ str(accuracy))
    print("Precision: " +str(precision))
    print("Recall: " +str(recall))
    print("F1-score: " +str(f1_Score))

if __name__ == '__main__':
    main()
