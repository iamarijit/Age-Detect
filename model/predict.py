from keras.models import load_model
from dataprocess import *
import imageprocess as ip
import numpy as np
import pandas as pd

xTest = ip.imgprocess(testImg,testDir)/255
model = load_model('../trainModel.h5')

yTest = model.predict(xTest)
#print(xTest.shape)
yTest = pd.DataFrame(yTest)
mTest = [max(i) for i in np.array(yTest)]

#print(yTest.head())
#print(mTest[:5])
#print(yTest.head()==mTest[:5])

yTest = yTest==mTest
yTest = yTest.astype('int')
yTest.columns = ['MIDDLE','OLD','YOUNG']
Class = yTest.idxmax(axis=1)
ID = pd.Series(testImg)
submission = pd.concat([Class,ID],axis=1)
submission.columns = ['Class','ID']

#print(Class.head())
#print(Class.shape)
#print(ID.head())
#print(ID.shape)
#print(submission.head())

submission.to_csv('../prediction.csv',index=False)
