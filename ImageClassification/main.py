
#import tensorflow
'''
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
'''
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt


# re-size all the images to this
IMAGE_SIZE = [224, 224]

def Preprocessing_Images():
    train_path = "../Images/seven_plastics/train"
    test_path = "../Images/seven_plastics/test"

    x_train = []

    for folder in os.listdir(train_path):
        sub_path = train_path + "/" + folder

        for img in os.listdir(sub_path):
            image_path = sub_path + "/" + img
            img_arr = cv2.imread(image_path)
            img_arr = cv2.resize(img_arr, (224, 224))
            x_train.append(img_arr)

    x_test = []

    for folder in os.listdir(test_path):
        sub_path = test_path + "/" + folder

        for img in os.listdir(sub_path):
            image_path = sub_path + "/" + img
            img_arr = cv2.imread(image_path)
            img_arr = cv2.resize(img_arr, (224, 224))
            x_test.append(img_arr)

    train_x = np.array(x_train)
    test_x = np.array(x_test)

    print(train_x.shape)
    print(test_x.shape)

    train_x = train_x / 255.0
    test_x = test_x / 255.0

    # train_datagen = ImageDataGenerator(rescale = 1./255,
    #                                    shear_range = 0.2,
    #                                    zoom_range = 0.2,
    #                                    horizontal_flip = True)

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(224, 224),
                                                     batch_size=32,
                                                     class_mode='sparse')
    test_set = test_datagen.flow_from_directory(test_path,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='sparse')

    train_y = training_set.classes
    test_y = test_set.classes

    print(training_set.class_indices)

    print(train_y.shape)
    print(test_y.shape)


def Model_Training():
    vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # do not train the pre-trained layers of VGG-19
    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.output)
    # adding output layer.Softmax classifier is used as it is multi-class classification
    prediction = Dense(3, activation='softmax')(x)

    model = Model(inputs=vgg.input, outputs=prediction)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    # view the structure of the model
    print(model.summary())


if __name__ == '__main__':
    Preprocessing_Images()


