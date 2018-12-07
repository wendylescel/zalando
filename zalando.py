# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 08:28:55 2018

@author: personnel
"""

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K 
K.set_image_dim_ordering('th')


# load the data (fashion_mnist)
(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

x_train=x_train.reshape(x_train.shape[0],1,28,28).astype('float32')
x_test=x_test.reshape(x_test.shape[0],1,28,28).astype('float32')

x_train=x_train/255
x_test=x_test/255
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]

def baseline_model():
    model=Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(1,28,28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Conv2D(64,(5,5),input_shape=(32,14,14), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    #model.add(Conv2D(8,(2,2),input_shape=(1,28,28), activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.4))
    #model.add(Dense(10,activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 
    
model=baseline_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15, batch_size=200, verbose=2)
scores=model.evaluate(x_test,y_test, verbose=0)
print("CNN error : %.2f%%" % (100-scores[1]*100))


