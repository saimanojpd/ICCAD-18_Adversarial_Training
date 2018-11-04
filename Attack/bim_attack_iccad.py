################################# Importing the libraries #######################################
import numpy as np
from keras.datasets import mnist
import keras
import matplotlib.pyplot as plt
from keras import backend
from keras.models import load_model
import tensorflow as tf
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras.optimizers import Adam

##################################### Trained Model ############################################
backend.set_learning_phase(False)
keras_model=load_model('/home/labadmin/Desktop/iccad_2018/fgsm_retrain/simple_nn_iccad.h5')

################################## Importing the dataset #######################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

############################ Set TF random seed to improve reproducibility ######################
tf.set_random_seed(1234)

if not hasattr(backend, "tf"):
    raise RuntimeError("This tutorial requires keras to be configured"
                       " to use the TensorFlow backend.")

if keras.backend.image_dim_ordering() != 'tf':
    keras.backend.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

#Definning the session
sess =  backend.get_session()

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None, 10))

pred = np.argmax(keras_model.predict(x_test), axis = 1)
acc =  np.mean(np.equal(pred, y_test))

print("The Test accuracy is: {}".format(acc))

#################################### Adversarial Attack (BIM) ###################################
wrap=KerasModelWrapper(keras_model)
bim = BasicIterativeMethod(wrap, sess=sess)
bim_params = {'eps': 0.9,
              'eps_iter': 0.6,
              'nb_iter': 10,
               'ord':np.inf,
              'clip_min': 0.,
              'clip_max': 1.}
adv_x = bim.generate_np(x_test, **bim_params)
adv_conf = keras_model.predict(adv_x)
adv_pred = np.argmax(adv_conf, axis = 1)
adv_acc =  np.mean(np.equal(adv_pred, y_test))

print("The adversarial  accuracy is: {}".format(adv_acc))

###################################### Original Image ##########################################
x_sample = x_test[7010].reshape(28, 28)
plt.imshow(x_sample,cmap='Blues')
plt.show()

###################################### Adversarial Image ########################################
adv_x_sample = adv_x[7010].reshape(28, 28)
plt.imshow(adv_x_sample,cmap='Blues')
plt.show()

########################################## Plotting ############################################
def stitch_images(images, y_img_count, x_img_count, margin = 2):
    
    # Dimensions of the images
    img_width = images[0].shape[0]
    img_height = images[0].shape[1]
    
    width = y_img_count * img_width + (y_img_count - 1) * margin
    height = x_img_count * img_height + (x_img_count - 1) * margin
    stitched_images = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(y_img_count):
        for j in range(x_img_count):
            img = images[i * x_img_count + j]
            if len(img.shape) == 2:
                img = np.dstack([img] * 3)
            stitched_images[(img_width + margin) * i: (img_width + margin) * i + img_width,
                            (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    return stitched_images

x_sample = x_test[4].reshape(28, 28)
adv_x_sample = adv_x[4].reshape(28, 28)

#Comparision
adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2)
plt.imshow(adv_comparison)
plt.show()

################################### Prediction of Neural Network ################################
#load the model we saved
model = load_model('/home/labadmin/Desktop/iccad_2018/fgsm_retrain/simple_nn_iccad.h5')
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

#Predicting the  Original image
test_image = x_test[7010]
classes = model.predict_classes(x_test[7010:7011], batch_size=10)
print (classes)

#Predicting the Adversarial Image
test_image = adv_x[7010]
classes = model.predict_classes(adv_x[7010:7011], batch_size=10)
print (classes)

################################### Saving adversarial inputs (TEST DATA)###################################
import pickle
filename = '/home/labadmin/Desktop/iccad_2018/fgsm_retrain/results_file/fgsm_train_bim_5'
pickle.dump(adv_x, open(filename, 'wb'))
 
# load the inputs from file
loaded_adv_x_test = pickle.load(open(filename, 'rb'))
