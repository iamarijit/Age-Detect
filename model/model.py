#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
#from keras.layers import Dense, Dropout, Flatten, BatchNormalization

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model


def model(inputShape = (28,28,3)):

    xInput = Input(inputShape)

    x = Conv2D(32,(3,3),strides = (1,1),name='conv0')(xInput)

    x = Conv2D(64,(3,3),strides = (1,1),name='conv1')(x)
    x = BatchNormalization(axis = 3,name = 'bn0')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2),name='maxPool0')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128,(3,3),strides = (1,1),name='conv2')(x)
    x = BatchNormalization(axis = 3,name = 'bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2),name='maxPool1')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256,(3,3),strides = (1,1),name='conv3')(x)
    x = BatchNormalization(axis = 3,name = 'bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2),name='maxPool2')(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128,activation = 'relu',name='fc0')(x)
    x = Dropout(0.5)(x)
    x = Dense(64,activation = 'relu',name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(3,activation = 'sigmoid',name='fc2')(x)

    model = Model(inputs = xInput,outputs = x,name='agePredict')

    return model
