#import library
import os, sys
from os import listdir
import h5py.defs
import h5py.utils
import h5py.h5ac
import h5py._proxy
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import shuffle
import PIL
from PIL import Image

#Set callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(((logs.get('val_accuracy') >= logs.get('accuracy')) and (logs.get('val_accuracy') >= 0.98) and (logs.get('val_loss') < 3) ) or ((logs.get('val_accuracy') >= 0.98) and (logs.get('val_loss') < 3))):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True
        if logs.get('val_accuracy') >= 0.7 and logs.get('val_accuracy') < 0.75 and (logs.get('val_loss') < 3) :
            if os.path.exists('D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_70.h5'):
                print('\n 70% model exists')
            else:
                model.save("D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_70.h5")
        if logs.get('val_accuracy') >= 0.75 and logs.get('val_accuracy') < 0.8 and (logs.get('val_loss') < 3):
            if os.path.exists('D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_75.h5'):
                print('\n 75% model exists')
            else:
                model.save("D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_75.h5")
        if logs.get('val_accuracy') >= 0.8 and logs.get('val_accuracy') < 0.85 and (logs.get('val_loss') < 3):
            if os.path.exists('D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_80.h5'):
                print('\n 80% model exists')
            else:
                model.save("D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_80.h5")
        if logs.get('val_accuracy') >= 0.85 and logs.get('val_accuracy') < 0.9 and (logs.get('val_loss') < 3):
            if os.path.exists('D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_85.h5'):
                print('\n 85% model exists')
            else:
                model.save("D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_85.h5")
        if logs.get('val_accuracy') >= 0.9 and logs.get('val_accuracy') < 0.95 and (logs.get('val_loss') < 3):
            if os.path.exists('D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_90.h5'):
                print('\n 90% model exists')
            else:
                model.save("D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_90.h5")
        if logs.get('val_accuracy') >= 0.95 and logs.get('val_accuracy') < 0.98 and (logs.get('val_loss') < 3):
            if os.path.exists('D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_95.h5'):
                print('\n 95% model exists')
            else:
                model.save("D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_95.h5")
                

callbacks = myCallback()

#Set image size and number of class
'''
drug_model_0 is Xception img_size = 299
drug_model_1 is RasNet152V2 img_size = 224
drug_model_2 is DenseNet201 img_size = 224
'''

img_size = 299
n_classes = 58

#set drug data path

drug_data_path = 'D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData'
train_dir = os.path.join(drug_data_path, 'train')
validate_dir = os.path.join(drug_data_path, 'validate')
data_train = ImageDataGenerator(rescale = np.divide(1,255.))
data_validate = ImageDataGenerator(rescale = np.divide(1,255.))
train_data_gen = data_train.flow_from_directory(train_dir, target_size = (img_size, img_size), batch_size = 8, class_mode = 'categorical' )
validate_data_gen = data_validate.flow_from_directory(validate_dir, target_size = (img_size, img_size), batch_size = 8, class_mode = 'categorical' )

#Select using model

#base_model = tf.keras.applications.ResNet152V2(input_shape = (img_size,img_size,3), include_top = False, weights = "imagenet")
base_model = tf.keras.applications.Xception(input_shape = (img_size,img_size,3), include_top = False, weights = "imagenet")
#base_model = tf.keras.applications.DenseNet201(input_shape = (img_size,img_size,3), include_top = False, weights = "imagenet")
base_model.summary()
base_model.trainable = False #train without adjust weight of node
avg_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
prediction_layer = tf.keras.layers.Dense(units = n_classes, activation = 'softmax')(avg_pooling_layer)
model = tf.keras.models.Model(inputs = base_model.input, outputs = prediction_layer)
model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# if run in jupyter or colab shoule change number of epochs to be 2 and run until get result you want
model.fit(train_data_gen, epochs=50, verbose=1, callbacks=None, validation_data=validate_data_gen, validation_steps=3)
model.save("D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_noChange_base.h5")

len(base_model.layers) #check number of layer

num_of_trainable_layer = 50 #this value should not higher than 30 percent of len(base_model.layers) and type of variable must be intreger

base_model.trainable = True
for layer in base_model.layers[:num_of_trainable_layer]:
    layer.trainable = False

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_gen, epochs=100, verbose=1, callbacks=[callbacks], validation_data=validate_data_gen, validation_steps=3)
model.save("D:/Users/T.CHUTIWAT/Desktop/AWS/KMUTT/Packing Machine/control/DrugData/model/drug_model_Change_base.h5")


