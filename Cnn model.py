import matplotlib.pyplot as plt
import tensorflow.keras as keras

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding
from keras.utils import to_categorical
from keras.layers import input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D,MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
model = Sequential()
#first layer
model.add(Conv1D(64, 5,padding='same',input_shape=(162,1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
#second layer
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
#third layer
model.add(Conv1D(256, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
#dense layer
model.add(Flatten())
model.add(Dense(8))
model.add(Activation('softmax'))
opt = keras.optimizers.Adaam(lr=0.00005)
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
cnnhistory=model.fit(x_traincnn,y_train,epochs=100,validation_data=(x_testcnn,ytest))
model.save('cnn_model.h5')