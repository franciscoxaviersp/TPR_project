import argparse
import os

import numpy as np
import scalogram
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from random import random


def slidingObsWindow(data, lengthObsWindow, slidingValue):
    windows = []
    nSamples, nMetrics = data.shape
    # print("Observation window size: {}\nSliding value: {}".format(lengthObsWindow, slidingValue))
    for s in np.arange(lengthObsWindow, nSamples, slidingValue):
        # print("\nAt sample: {}\n".format(s - 1))
        subdata = data[s - lengthObsWindow:s, :]
        # print(subdata)
        windows.append(subdata)
    return windows


def extractFeatures(data):
    features = []
    nObs, nSamp, nCols = data.shape
    for i in range(nObs):
        M1 = np.mean(data[i, :, :], axis=0)
        Std1 = np.std(data[i, :, :], axis=0)
        p = [75, 90, 95]
        Pr1 = np.array(np.percentile(data[i, :, :], p, axis=0)).T.flatten()

        a = np.sum(data[i, :, 0])
        b = np.sum(data[i, :, 2])
        total = a + b
        upload_ratio = 0
        download_ratio = 0
        # print(packet_ratio)
        if total != 0:
            upload_ratio = a / total
            download_ratio = b / total

        # faux=np.hstack((M1,Md1,Std1,S1,K1,Pr1))
        faux = np.hstack((M1, Std1, Pr1, np.array([upload_ratio, download_ratio])))
        features.append(faux)

    return (np.array(features))


def extratctSilence(data, threshold=128):
    if (data[0] <= threshold):
        s = [1]
    else:
        s = []
    for i in range(1, len(data)):
        if (data[i - 1] > threshold and data[i] <= threshold):
            s.append(1)
        elif (data[i - 1] <= threshold and data[i] <= threshold):
            s[-1] += 1

    return (s)


def extractFeaturesSilence(data):
    features = []
    nObs, nSamp, nCols = data.shape
    for i in range(nObs):
        silence_features = np.array([])
        for c in [0, 2]:
            silence = extratctSilence(data[i, :, c], threshold=0)
            if len(silence) > 0:
                silence_features = np.append(silence_features, [np.mean(silence), np.var(silence)])
            else:
                silence_features = np.append(silence_features, [0, 0])

        features.append(silence_features)

    return (np.array(features))

def extractFeaturesWavelet(data,scales=[2,4,8,16,32],Class=0):
    features=[]
    nObs,nSamp,nCols=data.shape
    for i in range(nObs):
        scalo_features=np.array([])
        for c in range(nCols):
            scalo,fscales=scalogram.scalogramCWT(data[i,:,c],scales)
            scalo_features=np.append(scalo_features,scalo)
            
        features.append(scalo_features)
        
    return np.array(features)

def main():
    windows_all = np.array([])
    i = 0
    nFiles = len(os.listdir('streams')) + len(os.listdir('streams05'))
    testFiles = []
    for folder in ['streams', 'streams05']:
        for file in os.listdir(folder):
            
            prob = random()
            if prob < 0.7:
                print(file)
                fileInput = open('{}/{}'.format(folder, file), 'r')
                data = np.loadtxt(fileInput, dtype=int)
                windows = np.array(slidingObsWindow(data, 3, 1))
                if i == 0:
                    windows_all = windows
                else:
                    windows_all = np.concatenate((windows, windows_all))
                i += 1
            else:
                testFiles.append('{}/{}'.format(folder, file))

    pca = PCA(n_components=3, svd_solver='full')

    features_timedependent = extractFeatures(windows_all)
    features_timeindependent = extractFeaturesSilence(windows_all)
    #featuresW = extractFeaturesWavelet(windows_all)
    train_features = np.hstack((features_timedependent, features_timeindependent))
    trainScaler = MaxAbsScaler().fit(train_features)
    train_featuresN=trainScaler.transform(train_features)
    trainPCA=pca.fit(train_featuresN)
    train_featuresNPCA = trainPCA.transform(train_featuresN)

    model = IsolationForest(n_estimators=50, max_samples='auto', contamination='auto', max_features=1.0)
    model.fit(train_featuresNPCA)

    totalAnom = 0
    totalTests = 0
    sizeAnom = 0
    sizeOk = 0

    for file in testFiles:
        print(file)
        test = np.loadtxt(open(file, 'r'), dtype=int)
        windows = np.array(slidingObsWindow(test, 3, 1))
        features_timedependent = extractFeatures(windows)
        features_timeindependent = extractFeaturesSilence(windows)
        #featuresW = extractFeaturesWavelet(windows)
        test_features = np.hstack((features_timedependent, features_timeindependent))

        test_featuresN=trainScaler.transform(test_features)
        test_featuresNPCA = trainPCA.transform(test_featuresN)
        #print(features)

        a = model.predict(test_featuresNPCA)
        b = model.decision_function(test_featuresNPCA)
        #print(a)
        #print(b)
        totalTests += 1
        anom = 0
        if len(test)-3>30:
            for prediction in a:
                if prediction == -1:
                    anom += 1
            
            if anom/len(a) >0.5:
                totalAnom += 1
                sizeAnom += len(test)-3
                print(a)
                print(b)
                print("ANOMALY")
            else:
                sizeOk += len(test)-3
                print("OK")

    test = np.loadtxt(open('streams2/3.txt', 'r'), dtype=int)
    windows = np.array(slidingObsWindow(test, 3, 1))
    features_timedependent = extractFeatures(windows)
    features_timeindependent = extractFeaturesSilence(windows)
    #featuresW = extractFeaturesWavelet(windows)
    test_features = np.hstack((features_timedependent, features_timeindependent))

    test_featuresN=trainScaler.transform(test_features)
    test_featuresNPCA = trainPCA.transform(test_featuresN)
    #print(features)

    a = model.predict(test_featuresNPCA)
    b = model.decision_function(test_featuresNPCA)
    print(a)
    print(b)

    print("OK average stream time (seconds): "+ str(sizeOk/(totalTests-totalAnom)))
    print("Anomalies average stream time (seconds): "+ str(sizeAnom/totalAnom))

    print("Test size: "+str(totalTests))
    print("Total anomalies: "+str(totalAnom))

    print("Percentage of accurate prediction on test dataset: "+str(100-(totalAnom/totalTests)*100))


if __name__ == '__main__':
    main()
