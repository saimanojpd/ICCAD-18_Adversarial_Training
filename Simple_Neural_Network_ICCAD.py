############################################# Importing the libraries########################################
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

############################################### Parameter Tunning ############################################
batch_size = 128
num_classes = 10
epochs =20
learning_rate=0.001

####################################### Spilting the data into test and training test #########################
(x_train, y_train), (x_test, y_test) = mnist.load_data()

########################################## Reshaping the data ################################################
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

################################################### Rescaling  ################################################
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

##################################### Convert class vectors to binary class matrices #########################
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

####################################### Initializing the Neural Network #######################################
model = Sequential()

# Adding the input layer and the 1st hidden layer 
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))

# Adding the 2nd hidden layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(num_classes, activation='softmax'))

############################################ Model Summary ###################################################
model.summary()

############################################## Training #######################################################
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=learning_rate),metrics=['accuracy'])

history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test))

################################################### Evaluating #################################################
score = model.evaluate(x_test, y_test, verbose=0) #TEST
print('Test loss:', score[0])
print('Test accuracy:', score[1])

score = model.evaluate(x_train, y_train, verbose=0) #TRAIN
print('Train loss:', score[0])
print('Train accuracy:', score[1])

################################################# Visualizing ###############################################################
# List all data in history
print(history.history.keys())
# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

################################################ Saving the model #############################################
model.save('/home/labadmin/Desktop/iccad_2018/fgsm_retrain/simple_nn_iccad.h5') 

################################################ Evaluation metrics ###################################################
from keras import backend
backend.set_learning_phase(False)
keras_model=load_model('/home/labadmin/Desktop/iccad_2018/fgsm_retrain/simple_nn_iccad.h5')
pred = np.argmax(keras_model.predict(x_test,None), axis = 1)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test.argmax(axis=1), pred)
from sklearn.metrics import classification_report
report = classification_report(y_test.argmax(axis=1), pred)
print(report)

########################################## Prediction of Neural Network #######################################
#load the model we saved
model = load_model('/home/labadmin/Desktop/iccad_2018/fgsm_retrain/simple_nn_iccad.h5')
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

#Predicting the  Original image
test_image = x_test[0]
classes = model.predict(x_test[0:1], batch_size=10)
print (classes)  

########################################### Prediction on adversarial samples #################################
from keras import backend
backend.set_learning_phase(False)
model=load_model('/home/labadmin/Desktop/iccad_2018/fgsm_retrain/simple_nn_iccad.h5')

import pickle
filename ='/home/labadmin/Desktop/iccad_2018/fgsm_retrain/results_file/fgsm_train_cw_1'
load_adversarial_samples= pickle.load(open(filename, 'rb'))

#Predicting the  Original image
test_image = load_adversarial_samples[0]
classes = model.predict_classes(load_adversarial_samples[0:1], batch_size=10)
print (classes) 


#Predicting the  Original image
test_image = load_adversarial_samples[0]
classes = model.predict(load_adversarial_samples[0:1], batch_size=10)
print (classes) 
 
 
score = model.evaluate(load_adversarial_samples ,y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])