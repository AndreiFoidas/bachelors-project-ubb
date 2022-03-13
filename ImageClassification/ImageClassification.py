#import tensorflow

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3
import efficientnet.tfkeras as efn

import numpy as np
import datetime
import os
import cv2
import matplotlib.pyplot as plt

# constants
IMAGE_SIZE = [224, 224]
EPOCHS = 10
PLOT = True
PRINT = True


class ImageClassification:
    def __init__(self):
        self._train_path = "../Images/seven_plastics/train"
        self._validator_path = "../Images/seven_plastics/test"
        self._test_path = "../Images/seven_plastics/manual_test"

        self._checkpoint_inception_path = "models/justTest.ckpt"
        self._checkpoint_vgg19_path = "models/vgg19cp1.ckpt"
        self._checkpoint_effnet_path = "models/effnetcp1.ckpt"
        self._checkpoint_dir = "models"
        self._file_dir = "files"

        self._train_x = None
        self._train_y = None
        self.Read_Images()

        self._efficientNet_model = None
        self._vgg19_model = None
        self.init_Vgg19()
        self.init_EfficientNet()

    def Read_Images(self):
        x_train = []
        x_validator = []

        for folder in os.listdir(self._train_path):
            sub_path = self._train_path + "/" + folder

            for img in os.listdir(sub_path):
                image_path = sub_path + "/" + img
                img_arr = cv2.imread(image_path)
                img_arr = cv2.resize(img_arr, IMAGE_SIZE)
                x_train.append(img_arr)

        for folder in os.listdir(self._validator_path):
            sub_path = self._validator_path + "/" + folder

            for img in os.listdir(sub_path):
                image_path = sub_path + "/" + img
                img_arr = cv2.imread(image_path)
                img_arr = cv2.resize(img_arr, IMAGE_SIZE)
                x_validator.append(img_arr)

        self._train_x = np.array(x_train)
        self._validator_x = np.array(x_validator)

        if PRINT:
            print(self._train_x.shape)
            print(self._validator_x.shape)

        self._train_x = self._train_x / 255.0
        self._validator_x = self._validator_x / 255.0

    def init_Vgg19(self):
        # def Model_Training(): #-------------------------------------------------------------------
        vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

        # do not train the pre-trained layers of VGG-19
        for layer in vgg.layers:
            layer.trainable = False

        x = Flatten()(vgg.output)
        # adding output layer.Softmax classifier is used as it is multi-class classification
        prediction = Dense(8, activation='softmax')(x)

        self._vgg19_model = Model(inputs=vgg.input, outputs=prediction)

        self._vgg19_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer="adam",
            metrics=['accuracy']
        )

    def init_EfficientNet(self):
        base_model = efn.EfficientNetB7(input_shape=IMAGE_SIZE + [3], include_top=False, weights='imagenet', classes=8)

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)

        # Add a final sigmoid layer with 1 node for classification output
        predictions = Dense(8, activation="sigmoid")(x)
        self._efficientNet_model = Model(base_model.input, predictions)

        self._efficientNet_model.compile(optimizer=RMSprop(lr=0.0001, decay=1e-6),
                                         loss='sparse_categorical_crossentropy',
                                         metrics=['accuracy'])

    def Train_VGG19(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2,
                                           height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                           horizontal_flip=True)
        validator_datagen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory(self._train_path,
                                                         target_size=IMAGE_SIZE,
                                                         batch_size=32,
                                                         class_mode='sparse')
        validator_set = validator_datagen.flow_from_directory(self._validator_path,
                                                              target_size=IMAGE_SIZE,
                                                              batch_size=32,
                                                              class_mode='sparse')

        train_y = training_set.classes
        validator_y = validator_set.classes

        if PRINT:
            print(training_set.class_indices)
            print(train_y.shape)
            print(validator_y.shape)

        # view the structure of the model
        if PRINT:
            print(self._vgg19_model.summary())

        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        # Early stopping to avoid overfitting of model
        cp_callback = ModelCheckpoint(filepath=self._checkpoint_vgg19_path, save_weights_only=True, verbose=1)
        # Create a callback that saves the model's weights

        # fit the model
        history = self._vgg19_model.fit(
            self._train_x,
            train_y,
            validation_data=(self._validator_x, validator_y),
            epochs=EPOCHS,
            callbacks=[early_stop, cp_callback],
            batch_size=32, shuffle=True)

        if PLOT:
            # accuracies
            plt.plot(history.history['accuracy'], label='train acc')
            plt.plot(history.history['val_accuracy'], label='val acc')
            plt.legend()
            plt.savefig(self._file_dir + '/vgg-acc-' + str(datetime.date.today()) + '.png')
            plt.show()

            # loss
            plt.plot(history.history['loss'], label='train loss')
            plt.plot(history.history['val_loss'], label='val loss')
            plt.legend()
            plt.savefig(self._file_dir + '/vgg-loss-' + str(datetime.date.today()) + '.png')
            plt.show()

    def Train_Inception(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255., rotation_range=40, width_shift_range=0.2,
                                           height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                           horizontal_flip=True)

        validator_datagen = ImageDataGenerator(rescale=1.0 / 255.)

        train_generator = train_datagen.flow_from_directory(self._train_path, batch_size=20, class_mode='sparse',
                                                            target_size=IMAGE_SIZE)
        validation_generator = validator_datagen.flow_from_directory(self._validator_path, batch_size=20,
                                                                     class_mode='sparse',
                                                                     target_size=IMAGE_SIZE)

        base_model = InceptionV3(input_shape=IMAGE_SIZE + [3], include_top=False, weights='imagenet', classes=8)

        for layer in base_model.layers:
            layer.trainable = False

        x = layers.Flatten()(base_model.output)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Add a final sigmoid layer with 1 node for classification output
        x = layers.Dense(8, activation='sigmoid')(x)

        model = Model(base_model.input, x)

        model.compile(optimizer=RMSprop(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['acc'])

        if PRINT:
            print(model.summary())

        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        # Early stopping to avoid overfitting of model
        cp_callback = ModelCheckpoint(filepath=self._checkpoint_inception_path, save_weights_only=True, verbose=1)
        # Create a callback that saves the model's weights

        history = model.fit_generator(train_generator, validation_data=validation_generator, steps_per_epoch=20,
                                      callbacks=[early_stop, cp_callback],
                                      epochs=EPOCHS)

        if PLOT:
            # accuracies
            plt.plot(history.history['accuracy'], label='train acc')
            plt.plot(history.history['val_accuracy'], label='val acc')
            plt.legend()
            plt.savefig(self._file_dir + '/inception-acc-' + str(datetime.date.today()) + '.png')
            plt.show()

            # loss
            plt.plot(history.history['loss'], label='train loss')
            plt.plot(history.history['val_loss'], label='val loss')
            plt.legend()
            plt.savefig(self._file_dir + '/inception-loss-' + str(datetime.date.today()) + '.png')
            plt.show()

    def Train_EfficientNet(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255.)

        validator_datagen = ImageDataGenerator(rescale=1.0 / 255.)

        train_generator = train_datagen.flow_from_directory(self._train_path, batch_size=20, class_mode='sparse',
                                                            target_size=IMAGE_SIZE)

        validation_generator = validator_datagen.flow_from_directory(self._validator_path, batch_size=20,
                                                                     class_mode='sparse',
                                                                     target_size=IMAGE_SIZE)

        train_y = train_generator.classes
        validator_y = validation_generator.classes

        if PRINT:
            print(self._efficientNet_model.summary())

        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        # Early stopping to avoid overfitting of model
        cp_callback = ModelCheckpoint(filepath=self._checkpoint_effnet_path, save_weights_only=True, verbose=1)
        # Create a callback that saves the model's weights

        history = self._efficientNet_model.fit_generator(train_generator,
                                                         validation_data=validation_generator,
                                                         steps_per_epoch=20,
                                                         callbacks=[early_stop, cp_callback],
                                                         epochs=EPOCHS)

        if PLOT:
            # accuracies
            plt.plot(history.history['accuracy'], label='train acc')
            plt.plot(history.history['val_accuracy'], label='val acc')
            plt.legend()
            plt.savefig(self._file_dir + '/effnet-acc-' + str(datetime.date.today()) + '.png')
            plt.show()

            # loss
            plt.plot(history.history['loss'], label='train loss')
            plt.plot(history.history['val_loss'], label='val loss')
            plt.legend()
            plt.savefig(self._file_dir + '/effnet-loss-' + str(datetime.date.today()) + '.png')
            plt.show()

    def Test_EfficientNet(self):
        self._efficientNet_model.load_weights(self._checkpoint_effnet_path)

        for img in os.listdir(self._test_path):
            img_name = img
            img = image.load_img(self._test_path + "/" + img, target_size=(224, 224))
            plt.imshow(img)
            plt.show()
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            pred = self._efficientNet_model.predict(images, batch_size=1)
            print("Guessed val: " + str(img_name) + " " + str(pred))
            ct = 0
            for i in pred[0]:
                ct = ct + 1
                print(str(ct) + ' - {:.2f}%'.format(i * 100))

    def Test_VGG19(self):
        self._vgg19_model.load_weights(self._checkpoint_vgg19_path)

        for img in os.listdir(self._test_path):
            img_name = img
            img = image.load_img(self._test_path + "/" + img, target_size=(224, 224))
            plt.imshow(img)
            plt.show()
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            pred = self._vgg19_model.predict(images, batch_size=1)
            print("Guessed val: " + str(img_name) + " " + str(pred))
            ct = 0
            for i in pred[0]:
                ct = ct + 1
                print(str(ct) + ' - {:.2f}%'.format(i * 100))


if __name__ == '__main__':
    while True:
        mode = 0
        ic = ImageClassification()
        try:
            mode = int(input('Input:'))
        except ValueError:
            print("Not a number")
        if mode == 1:
            ic.Train_VGG19()
            ic.Test_VGG19()
        elif mode == 2:
            ic.Train_EfficientNet()
            ic.Test_EfficientNet()
        elif mode == 3:
            ic.Train_Inception()
        elif mode == 4:
            ic.Test_VGG19()
        elif mode == 5:
            ic.Test_EfficientNet()
        elif mode == 0:
            break
