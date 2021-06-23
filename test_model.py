import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MaxAbsScaler

from featureExtraction import (extractFeatures, extractFeaturesSilence,
                               slidingObsWindow)


def predict(folder, file_check=False, debug=False):

    global model
    global trainScaler
    global trainPCA
    global svm_model
    global model_auto

    totalAnom = 0
    totalSVMAnom = 0
    totalAnomAuto = 0
    totalSessions = 0

    features = []
    thresholds = []

    if file_check:
        test_files = open(folder, "r").read().split("\n")
    else:
        test_files = os.listdir(folder)

    for file in test_files:

        if file_check:
            test = np.loadtxt(open(file, 'r'), dtype=int)
        else:
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
        c = model_auto.predict(test_featuresNPCA)
        d = model_auto.decision_function(test_featuresNPCA)
        scores = svm_model.score_samples(test_featuresN)
        threshold = np.quantile(scores, 0.05)
        thresholds.append(threshold)
        threshold_calculated = 21987
        
        totalSessions += 1
        anom = 0
        anom_auto = 0
        svm_anom = 0
        if len(test)-3>15:
            for i,prediction in enumerate(a):
                if prediction == -1 and b[i] <= 0:
                    anom += 1
                if scores[i] <= threshold_calculated:
                    svm_anom += 1
            for i,prediction in enumerate(c):
                if prediction == -1 and d[i] <= 0:
                    anom_auto += 1


            
            if anom/len(a) >0.5:
                totalAnom += 1
                if debug:
                    print(a)
                    print(b)
                    print(np.average(b))
                    print("ANOMALY")
            else:
                if debug:
                    print(np.average(b))
                    print("OK")

            if svm_anom/len(a) >0.5:
                totalSVMAnom += 1
            if anom_auto/len(a) >0.5:
                totalAnomAuto += 1

    print(np.mean(np.array(thresholds, dtype=np.float32)))
    return (totalSessions,totalAnom, totalSVMAnom, totalAnomAuto)

def main():

    try:
        debug = sys.argv[1] == 'true'
    except:
        debug = False
    global model
    global trainScaler
    global trainPCA
    global svm_model
    global model_auto

    with open('scaler', 'rb') as pickle_file:
        trainScaler = pickle.load(pickle_file)
    with open('pca', 'rb') as pickle_file:
        trainPCA = pickle.load(pickle_file)
    with open('model', 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    with open('svm_model', 'rb') as pickle_file:
        svm_model = pickle.load(pickle_file)
    with open('model_auto', 'rb') as pickle_file:
        model_auto = pickle.load(pickle_file)


    print("Predicting on testing sessions")
    total_ok, anomalies, svm_anomalies, anom_auto = predict("test_files", file_check=True, debug=debug)
    print("SVM_Anomalies: " + str(svm_anomalies))
    print("IsolationForest_Anomalies: " + str(anomalies))
    print("IsolationForest_Anomalies_Default: " + str(anom_auto))


    ok = total_ok - anomalies
    FP = anomalies
    TN = ok
    print("True negatives: " + str(TN))
    print("False positives: " + str(FP))
    print("Percentage of false positives: " + str(FP / total_ok))

    #testing ok/non anomaly sessions
    print("Predicting on ok sessions")
    total_ok, anomalies, svm_anomalies, anom_auto = predict("streams_ok", debug=debug)
    print("SVM_Anomalies: " + str(svm_anomalies))
    print("IsolationForest_Anomalies: " + str(anomalies))
    print("IsolationForest_Anomalies_Default: " + str(anom_auto))


    ok = total_ok - anomalies
    FP = anomalies
    TN = ok
    print("True negatives: "+str(TN))
    print("False positives: "+str(FP))
    print("Percentage of false positives: "+str(FP/total_ok))

    #testing anomalous sessions
    
    print("Predicting on anomalous sessions")
    total_anom, anomalies, svm_anomalies, anom_auto = predict("streams_anom", debug=debug)
    print("SVM_Anomalies: " + str(svm_anomalies))
    print("IsolationForest_Anomalies: " + str(anomalies))
    print("IsolationForest_Anomalies_Default: " + str(anom_auto))

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
