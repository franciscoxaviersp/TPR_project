
import os
import pickle
from random import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import OneClassSVM


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


def main():
    windows_all = np.array([])
    i = 0
    testFiles = []
    for folder in ['streams_dataset_1', 'streams_dataset_2']:
        for file in os.listdir(folder):
            
            prob = random()
            if prob < 0.7:
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
    train_features = np.hstack((features_timedependent, features_timeindependent))
    trainScaler = MaxAbsScaler().fit(train_features)
    train_featuresN=trainScaler.transform(train_features)
    trainPCA=pca.fit(train_featuresN)
    train_featuresNPCA = trainPCA.transform(train_featuresN)

    model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1), max_features=1.0)
    model_auto = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0)
    model.fit(train_featuresNPCA)
    model_auto.fit(train_featuresNPCA)

    with open("scaler","wb") as f:
        pickle.dump(trainScaler,f)
        f.close()
    with open("pca","wb") as f:
        pickle.dump(trainPCA,f)
        f.close()
    with open("model","wb") as f:
        pickle.dump(model,f)
        f.close()
    with open("model_auto","wb") as f:
        pickle.dump(model_auto,f)
        f.close()
    with open("test_files", "w") as f:
        for file in testFiles:
            f.write(file+"\n")

    #One class SVM

    svm = OneClassSVM(kernel="linear")
    svm.fit(train_featuresN)

    with open("svm_model", 'wb') as f:
        pickle.dump(svm, f)


if __name__ == '__main__':
    main()
