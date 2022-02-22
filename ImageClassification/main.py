# import tensorflow

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224, 224]
train_path = "../Images/seven_plastics/train"
test_path = "../Images/seven_plastics/test"
res_path = "../Images/seven_plastics/manual_test"
#checkpoint_path = "models/cp.ckpt"
checkpoint_path = "models/cpF.ckpt"
#checkpoint_path = "models/cpGray.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

def Preprocessing_Images():

    x_train = []

    for folder in os.listdir(train_path):
        sub_path = train_path + "/" + folder

        for img in os.listdir(sub_path):
            image_path = sub_path + "/" + img
            img_arr = cv2.imread(image_path)
            img_arr = cv2.resize(img_arr, (224, 224))
            #img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            x_train.append(img_arr)

    x_test = []

    for folder in os.listdir(test_path):
        sub_path = test_path + "/" + folder

        for img in os.listdir(sub_path):
            image_path = sub_path + "/" + img
            img_arr = cv2.imread(image_path)
            img_arr = cv2.resize(img_arr, (224, 224))
            #img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            x_test.append(img_arr)

    train_x = np.array(x_train)
    test_x = np.array(x_test)

    #train_x = np.repeat(train_x[..., np.newaxis], 3, -1)
    #test_x = np.repeat(test_x[..., np.newaxis], 3, -1)

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

    # def Model_Training(): #-------------------------------------------------------------------
    vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # do not train the pre-trained layers of VGG-19
    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.output)
    # adding output layer.Softmax classifier is used as it is multi-class classification
    prediction = Dense(8, activation='softmax')(x)

    model = Model(inputs=vgg.input, outputs=prediction)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    # view the structure of the model
    print(model.summary())

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    # Early stopping to avoid overfitting of model

    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    # Create a callback that saves the model's weights

    # fit the model
    history = model.fit(
        train_x,
        train_y,
        validation_data=(test_x, test_y),
        epochs=15,
        callbacks=[early_stop, cp_callback],
        batch_size=32, shuffle=True)

    # accuracies
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.savefig('vgg-acc-rps-1.png')
    plt.show()

    # loss
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig('vgg-loss-rps-1.png')
    plt.show()

    os.listdir(checkpoint_dir)

    for img in os.listdir(res_path):
        img_name = img
        img = image.load_img(res_path + "/" + img, target_size=(224, 224))
        plt.imshow(img)
        plt.show()
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = model.predict(images, batch_size=1)
        print("Guessed val: " + str(img_name) + " " + str(pred))
        if pred[0][0] > 0.5:
            print("1 PET")
        elif pred[0][1] > 0.5:
            print("2 PE HD")
        elif pred[0][2] > 0.5:
            print("3 PVC")
        elif pred[0][3] > 0.5:
            print("4 PE LD")
        elif pred[0][4] > 0.5:
            print("5 PP")
        elif pred[0][5] > 0.5:
            print("6 PS")
        elif pred[0][6] > 0.5:
            print("7 OTHER")
        elif pred[0][7] > 0.5:
            print("8 NOT PLASTIC")
        else:
            print("Unknown")


def Testing_Images():
    # def Model_Training(): #-------------------------------------------------------------------
    vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # do not train the pre-trained layers of VGG-19
    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.output)
    # adding output layer.Softmax classifier is used as it is multi-class classification
    prediction = Dense(8, activation='softmax')(x)

    model = Model(inputs=vgg.input, outputs=prediction)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    model.load_weights(checkpoint_path)

    for img in os.listdir(res_path):
        img_name = img
        img = image.load_img(res_path + "/" + img, target_size=(224, 224))
        plt.imshow(img)
        plt.show()
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = model.predict(images, batch_size=1)
        print("Guessed val: " + str(img_name) + " " + str(pred))
        if pred[0][0] > 0.5:
            print("1 PET")
        elif pred[0][1] > 0.5:
            print("2 PE HD")
        elif pred[0][2] > 0.5:
            print("3 PVC")
        elif pred[0][3] > 0.5:
            print("4 PE LD")
        elif pred[0][4] > 0.5:
            print("5 PP")
        elif pred[0][5] > 0.5:
            print("6 PS")
        elif pred[0][6] > 0.5:
            print("7 OTHER")
        elif pred[0][7] > 0.5:
            print("8 NOT PLASTIC")
        else:
            print("Unknown")


if __name__ == '__main__':
    mode = 0
    try:
        mode = int(input('Input:'))
    except ValueError:
        print("Not a number")
    if mode == 1:
        Preprocessing_Images()
    if mode == 2:
        Testing_Images()
