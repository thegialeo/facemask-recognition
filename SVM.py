#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:01:47 2020

@author: simon.scholz
"""

import matplotlib.pyplot as plt

import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

import os
import cv2

XTrain = []
XTest = []
yTest = []
yTrain = []
trainPath = "../dataset/Mask_Datasets/Train"
smallPath = "../datasetSmall/Mask_Datasets/Train"
maskCount=0
noMaskCount=0
testMaskCount=0
testNoMaskCount=0

def initData():
    global XTrain 
    global XTest 
    global yTrain 
    global yTest 
    global trainPath 
    global maskCount
    global noMaskCount
    global testMaskCount
    global testNoMaskCount

    for folder in os.listdir(trainPath):
        if("." not in folder):
            for file in os.listdir(os.path.join(trainPath, folder)):
                if not file.startswith('.'):
                    #print("/"+file)
                    img = cv2.imread(os.path.join(trainPath, folder, file))
                    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255
                    XTrain.append(np.array(img))
                    if(folder.startswith('No')):
                       noMaskCount+=1
                    else:
                        maskCount+=1
                         
    XTrain = np.array(XTrain)
    yTrain = np.concatenate( (np.ones(maskCount), np.zeros(noMaskCount)), axis=0)
    
    
    testPath = "../dataset/Mask_Datasets/Validation"
    for folder in os.listdir(testPath):
        if("." not in folder):
            for file in os.listdir(os.path.join(testPath, folder)):
                if not file.startswith('.'):
                    # print("/"+file)
                    img = cv2.imread(os.path.join(testPath, folder, file))
                    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255
                    XTest.append(np.array(img))
                    if(folder.startswith('No')):
                       testNoMaskCount+=1
                    else:
                        testMaskCount+=1
                         
    XTest = np.array(XTest)
    yTest = np.concatenate( (np.ones(testMaskCount), np.zeros(testNoMaskCount)), axis=0)
    
    XTrain, yTrain = shuffle(XTrain, yTrain)
    XTrain = XTrain.reshape((XTrain.shape[0], -1))
    
    XTest, yTest = shuffle(XTest, yTest)
    XTest = XTest.reshape((XTest.shape[0], -1))
    print("init finished")


def calcSvc():
    global XTrain 
    global XTest 
    global trainPath 
    global maskCount
    global noMaskCount
    global testMaskCount
    global testNoMaskCount
    global yTrain 
    global yTest 
    
    ###  Support Vector Machine  ###
    
    kernel=['linear', 'poly', 'rbf', 'sigmoid']
    for k in kernel:
        print("kernel=" + k)
        clfLoop = svm.SVC(kernel = k, C = 1, probability=True)
        clfLoop.fit(XTrain, yTrain)
        scores = cross_val_score(clfLoop, XTest, yTest, cv = 3)
        
        print("Scores: ", scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("")
    

# def calcNearestNeighbour():
#     print("neighbour")
#     global XTrain 
#     global XTest 
#     global trainPath 
#     global maskCount
#     global noMaskCount
#     global testMaskCount
#     global testNoMaskCount
#     global yTrain 
#     global yTest 
    
#     clfLoop = NearestCentroid()
#     clfLoop.fit(XTrain, yTrain)
#     predicted = clfLoop.predict(XTest.reshape((XTest.shape[0], -1)))
#     print("red", predicted)
#     confusion_matrix(yTest, predicted)
#     disp = plot_confusion_matrix(clfLoop, XTest, yTest,
#                                  display_labels="test",
#                                  cmap=plt.cm.Blues,
#                                  )
#     print(disp.confusion_matrix)
    
    
#     clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1, 
#                                      max_depth=1, random_state=0).fit(XTrain, yTrain)
#     print("score: ", clf.score(XTest, yTest))
#     print("neighbours" )
#     print("Scores: ", scores)
#     print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#     print("")
    
    
  
    
initData()    
calcSvc()




#######
# _, axes = plt.subplots(2, 4)
# images_and_labels = list(zip(XTrain, yTrain))
# for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)

# images_and_predictions = list(zip(XTest, clf.predict(XTest.reshape((XTest.shape[0], -1)))))
# for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Prediction: %i' % prediction)


