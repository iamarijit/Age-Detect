import os
import pandas as pd

trainDir = "../train/"
testDir = "../test/"

#for img in os.listdir(trainDir):
#    trainImg.append(img)

#for img in os.listdir(testDir):
#    testImg.append(img)

trainData = pd.read_csv("../train.csv")
testData = pd.read_csv("../test.csv")
trainImg = list(trainData['ID'])
testImg = list(testData['ID'])
#no. of training examples
noTrainImages = len(trainImg)
#no. of test examples
noTestImages = len(testImg)

#one-hot-encode age class
encodeClass = pd.get_dummies(trainData['Class'])
trainData = trainData.join(encodeClass)
#print(trainData)
