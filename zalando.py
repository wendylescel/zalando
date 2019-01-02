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
    model.add(Conv2D(32,kernel_size=(3,3),input_shape=(1,28,28),kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D((2, 2)))  
    model.add(Dropout(0.25))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))  
    model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
   
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary();
    return model 
    
model=baseline_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=128, verbose=2)
scores=model.evaluate(x_test,y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("CNN error : %.2f%%" % (100-scores[1]*100))


