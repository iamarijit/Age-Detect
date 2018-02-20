from dataprocess import *
import imageprocess as ip
import cv2
import model
import numpy as np
from keras.models import Model
from keras.models import load_model

xTrain = ip.imgprocess(trainImg,trainDir)/255
yTrain = np.array(trainData[['MIDDLE','OLD','YOUNG']])

#print(type(xTrain))
#print(yTrain[:5])

'''
for i in range(11):
    print(xTrain[i].dtype)
    cv2.imshow(trainData['Class'][i], xTrain[i])
    cv2.waitKey(0)
'''

trainModel = model.model()
trainModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(trainModel.summary())

trainModel.fit(xTrain,yTrain,epochs = 150,batch_size = 1000)

preds = trainModel.evaluate(xTrain,yTrain)
print ("Loss = " + str(preds[0]))
print ("Train Accuracy = " + str(preds[1]))

trainModel.save('../trainModel.h5')
