# import tensorflow

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
import efficientnet.tfkeras as efn
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
from datetime import datetime
import os
import cv2
import matplotlib.pyplot as plt

# constants
IMAGE_SIZE = [224, 224]
EPOCHS = 25
PLOT = True
PRINT = True
NUM_FOLDS = 5


class ImageClassification:
    def __init__(self):
        self._train_path = "../UneditedImages/seven_plastics/train"
        self._validator_path = "../UneditedImages/seven_plastics/test"
        self._test_path = "../UneditedImages/seven_plastics/manual_test"

        # test accuracy
        self._all_photos_test_path = "../TestingImages"

        # k-cross validations
        self._all_photos_path = "../ImagesAllUnedited/seven_plastics"
        self._all_photos_path2 = "../ImagesAllUnedited/all_photos"
        self._dataframe = None

        self._checkpoint_vgg19_kfold_path = "models/vgg19_lr_kfoldcp"  # 1.ckpt
        self._checkpoint_effnet_kfold_path = "models/effnet_un_FT__kfoldcp"  # 1.ckpt
        self._all_data = None

        # checkpoints
        self._checkpoint_vgg19_path = "models/vgg19cp1.ckpt"
        self._checkpoint_effnet_path = "models/effnetcp1.ckpt"
        self._checkpoint_myModel_path = "models/myModelcp1.ckpt"

        self._good_checkpoint_vgg19_path = "goodmodels/vgg19_lr_kfoldcp1.ckpt"
        self._good_checkpoint_effnet_path = "goodmodels/effnet_un_FT__kfoldcp4.ckpt"

        self._checkpoint_dir = "models"
        self._file_dir = "files"

        self._train_x = None
        self._train_y = None
        # self.Read_Images()

        self._effnet_base_model = None
        self._vgg19_base_model = None

        self._efficientNet_model = None
        self._vgg19_model = None


        self.init_Vgg19()
        self.init_EfficientNet()


    def Load_Models(self):
        self._efficientNet_model.load_weights(self._good_checkpoint_effnet_path)
        self._vgg19_model.load_weights(self._good_checkpoint_vgg19_path)

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

    def Read_Images_Kfold(self):
        self._all_data = []
        data = []
        idx = 1

        for folder in os.listdir(self._all_photos_path):
            sub_path = self._train_path + "/" + folder

            for img in os.listdir(sub_path):
                image_path = sub_path + "/" + img
                img_arr = cv2.imread(image_path)
                img_arr = cv2.resize(img_arr, IMAGE_SIZE)
                self._all_data.append(img_arr)
                data.append([img, str(idx)])

            idx += 1

        self._all_data = np.array(self._all_data)
        self._dataframe = pd.DataFrame(data, columns=["id", "label"])
        if PRINT:
            print(self._all_data.shape)
            print(self._dataframe)

        self._all_data = self._all_data / 255.0

    def init_Vgg19(self):
        self._vgg19_base_model = VGG19(input_shape=IMAGE_SIZE + [3],
                                       weights='imagenet', include_top=False)

        #  not training the pre-trained layers of VGG-19
        for layer in self._vgg19_base_model.layers:
            layer.trainable = False

        x = Flatten()(self._vgg19_base_model.output)
        # adding an output layer; Softmax classifier is used as it
        # is multi-class classification
        prediction = Dense(8, activation='softmax')(x)

        self._vgg19_model = Model(inputs=self._vgg19_base_model.input, outputs=prediction)
        self._vgg19_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=RMSprop(learning_rate=0.0001, decay=1e-6),
            metrics=['accuracy']
        )

    def init_EfficientNet(self):
        self._effnet_base_model = efn.EfficientNetB7(input_shape=IMAGE_SIZE + [3],
                                                     include_top=False,
                                                     weights='imagenet', classes=8)

        #  not training the pre-trained layers of EfficientNet
        for layer in self._effnet_base_model.layers:
            layer.trainable = False

        x = self._effnet_base_model.output
        x = Flatten()(x)
        # Adding a final layer with 8 nodes for classification output
        predictions = Dense(8, activation="softmax")(x)

        self._efficientNet_model = Model(self._effnet_base_model.input, predictions)
        self._efficientNet_model.compile(optimizer=RMSprop(learning_rate=0.0001, decay=1e-6),
                                         loss='sparse_categorical_crossentropy',
                                         metrics=['accuracy'])

    def Train_VGG19_Kfold(self):
        df = self._dataframe.copy()
        kfold = StratifiedKFold(n_splits=NUM_FOLDS)
        idx = 0

        data_kfold = pd.DataFrame()

        train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=90, width_shift_range=0.2,
                                           height_shift_range=0.2, shear_range=0.2, zoom_range=0.2)

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        train_y = df.label
        train_x = df.drop(['label'], axis=1)

        for train_idx, val_idx in list(kfold.split(train_x, train_y)):
            idx += 1
            x_train_df = df.iloc[train_idx]
            x_valid_df = df.iloc[val_idx]
            # view the structure of the model
            if PRINT:
                print(self._vgg19_model.summary())

            training_set = train_datagen.flow_from_dataframe(dataframe=x_train_df,
                                                             directory=self._all_photos_path2,
                                                             x_col="id", y_col="label",
                                                             target_size=IMAGE_SIZE,
                                                             batch_size=32,
                                                             class_mode='sparse')
            validator_set = validation_datagen.flow_from_dataframe(dataframe=x_valid_df,
                                                                   directory=self._all_photos_path2,
                                                                   x_col="id", y_col="label",
                                                                   target_size=IMAGE_SIZE,
                                                                   batch_size=32,
                                                                   class_mode='sparse')

            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
            # Early stopping to avoid overfitting of model
            cp_callback = ModelCheckpoint(filepath=self._checkpoint_vgg19_kfold_path + str(idx) + ".ckpt",
                                          save_weights_only=True, verbose=1)
            # Create a callback that saves the model's weights

            # fit the model
            history = self._vgg19_model.fit(
                training_set,
                validation_data=validator_set,
                epochs=EPOCHS,
                callbacks=[early_stop, cp_callback],
                batch_size=32, shuffle=True)

            if PLOT:
                # accuracies
                plt.plot(history.history['accuracy'], label='train acc')
                plt.plot(history.history['val_accuracy'], label='val acc')
                plt.legend()
                plt.savefig(self._file_dir + '/vgg-acc-' + "_unedited_kfold" + str(idx) + "-"
                            + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
                plt.show()

                # loss
                plt.plot(history.history['loss'], label='train loss')
                plt.plot(history.history['val_loss'], label='val loss')
                plt.legend()
                plt.savefig(self._file_dir + '/vgg-loss-' + "_unedited_kfold" + str(idx) + "-"
                            + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
                plt.show()

    def Train_EfficientNet_Kfold(self):
        df = self._dataframe.copy()
        kfold = StratifiedKFold(n_splits=NUM_FOLDS)
        idx = 0

        data_kfold = pd.DataFrame()

        train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=90, width_shift_range=0.2,
                                           height_shift_range=0.2, shear_range=0.2, zoom_range=0.2)

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        train_y = df.label
        train_x = df.drop(['label'], axis=1)

        for train_idx, val_idx in list(kfold.split(train_x, train_y)):
            idx += 1
            x_train_df = df.iloc[train_idx]
            x_valid_df = df.iloc[val_idx]

            # view the structure of the model
            if PRINT:
                print(self._vgg19_model.summary())

            training_set = train_datagen.flow_from_dataframe(dataframe=x_train_df,
                                                             directory=self._all_photos_path2,
                                                             x_col="id", y_col="label",
                                                             target_size=IMAGE_SIZE,
                                                             batch_size=32,
                                                             class_mode='sparse')

            validator_set = validation_datagen.flow_from_dataframe(dataframe=x_valid_df,
                                                                   directory=self._all_photos_path2,
                                                                   x_col="id", y_col="label",
                                                                   target_size=IMAGE_SIZE,
                                                                   batch_size=32,
                                                                   class_mode='sparse')

            # view the structure of the model
            if PRINT:
                print(self._vgg19_model.summary())

            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
            # Early stopping to avoid overfitting of model
            cp_callback = ModelCheckpoint(filepath=self._checkpoint_effnet_kfold_path + str(idx) + ".ckpt",
                                          save_weights_only=True, verbose=1)
            # Create a callback that saves the model's weights

            history = self._efficientNet_model.fit(training_set,
                                                   validation_data=validator_set,
                                                   # steps_per_epoch=25,
                                                   callbacks=[early_stop, cp_callback],
                                                   epochs=EPOCHS)

            if PLOT:
                # accuracies
                plt.plot(history.history['accuracy'], label='train acc')
                plt.plot(history.history['val_accuracy'], label='val acc')
                plt.legend()
                plt.savefig(self._file_dir + '/effnet-acc-' + "_unedited_kfold" + str(idx) + "-"
                            + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
                plt.show()

                # loss
                plt.plot(history.history['loss'], label='train loss')
                plt.plot(history.history['val_loss'], label='val loss')
                plt.legend()
                plt.savefig(self._file_dir + '/effnet-loss-' + "_unedited_kfold" + str(idx) + "-"
                            + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
                plt.show()

    def Train_EfficientNet_Kfold_FineTune(self):
        df = self._dataframe.copy()
        kfold = StratifiedKFold(n_splits=NUM_FOLDS)
        idx = 0

        data_kfold = pd.DataFrame()

        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=90,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2)

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        train_y = df.label
        train_x = df.drop(['label'], axis=1)

        for train_idx, val_idx in list(kfold.split(train_x, train_y)):
            idx += 1
            x_train_df = df.iloc[train_idx]
            x_valid_df = df.iloc[val_idx]


            training_set = train_datagen.flow_from_dataframe(dataframe=x_train_df,
                                                             directory=self._all_photos_path2,
                                                             x_col="id", y_col="label",
                                                             target_size=IMAGE_SIZE,
                                                             batch_size=32,
                                                             class_mode='sparse')

            validator_set = validation_datagen.flow_from_dataframe(dataframe=x_valid_df,
                                                                   directory=self._all_photos_path2,
                                                                   x_col="id", y_col="label",
                                                                   target_size=IMAGE_SIZE,
                                                                   batch_size=32,
                                                                   class_mode='sparse')

            # view the structure of the model
            if PRINT:
                # print(_efficientNet_model.summary())
                pass

            # Early stopping to avoid overfitting of model
            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
            # CreatING a callback that saves the model's weights
            cp_callback = ModelCheckpoint(filepath=self._checkpoint_effnet_kfold_path + str(idx) + ".ckpt",
                                          save_weights_only=True, verbose=1)


            history = self._efficientNet_model.fit(training_set,
                                                   validation_data=validator_set,
                                                   # steps_per_epoch=25,
                                                   callbacks=[early_stop, cp_callback],
                                                   epochs=EPOCHS)

            if PLOT:
                # accuracies
                plt.plot(history.history['accuracy'], label='train acc')
                plt.plot(history.history['val_accuracy'], label='val acc')
                plt.legend()
                plt.savefig(self._file_dir + '/effnet-acc-' + "lr_kfold" + str(idx) + "-"
                            + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
                plt.show()

                # loss
                plt.plot(history.history['loss'], label='train loss')
                plt.plot(history.history['val_loss'], label='val loss')
                plt.legend()
                plt.savefig(self._file_dir + '/effnet-loss-' + "lr_kfold" + str(idx) + "-"
                            + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
                plt.show()

            # def Finetune_VGG19(self):

            # unfreeze layers for finetuning
            for layer in self._effnet_base_model.layers:
                layer.trainable = True

            self._efficientNet_model.compile(optimizer=RMSprop(learning_rate=0.000001, decay=1e-6),
                                             loss='sparse_categorical_crossentropy',
                                             metrics=['accuracy'])

            history = self._efficientNet_model.fit(training_set,
                                                   validation_data=validator_set,
                                                   callbacks=[early_stop, cp_callback],
                                                   epochs=EPOCHS // 2)

            if PLOT:
                # accuracies
                plt.plot(history.history['accuracy'], label='train acc')
                plt.plot(history.history['val_accuracy'], label='val acc')
                plt.legend()
                plt.savefig(self._file_dir + '/effnet-acc-' + "FT_kfold" + str(idx) + "-"
                            + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
                plt.show()

                # loss
                plt.plot(history.history['loss'], label='train loss')
                plt.plot(history.history['val_loss'], label='val loss')
                plt.legend()
                plt.savefig(self._file_dir + '/effnet-loss-' + "FT_kfold" + str(idx) + "-"
                            + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
                plt.show()

    def Train_VGG19(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=90, width_shift_range=0.2,
                                           height_shift_range=0.2, shear_range=0.2, zoom_range=0.2)

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
            plt.savefig(self._file_dir + '/vgg-acc-' + str(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
            plt.show()

            # loss
            plt.plot(history.history['loss'], label='train loss')
            plt.plot(history.history['val_loss'], label='val loss')
            plt.legend()
            plt.savefig(self._file_dir + '/vgg-loss-' + str(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
            plt.show()

    def Train_EfficientNet(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=90, width_shift_range=0.2,
                                           height_shift_range=0.2, shear_range=0.2, zoom_range=0.2)

        validator_datagen = ImageDataGenerator(rescale=1.0 / 255.)

        train_generator = train_datagen.flow_from_directory(self._train_path, batch_size=32, class_mode='sparse',
                                                            target_size=IMAGE_SIZE)

        validation_generator = validator_datagen.flow_from_directory(self._validator_path, batch_size=32,
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

        history = self._efficientNet_model.fit(train_generator,
                                               validation_data=validation_generator,
                                               steps_per_epoch=25,
                                               callbacks=[early_stop, cp_callback],
                                               epochs=EPOCHS)

        if PLOT:
            # accuracies
            plt.plot(history.history['accuracy'], label='train acc')
            plt.plot(history.history['val_accuracy'], label='val acc')
            plt.legend()
            plt.savefig(self._file_dir + '/effnet-acc-' + str(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
            plt.show()

            # loss
            plt.plot(history.history['loss'], label='train loss')
            plt.plot(history.history['val_loss'], label='val loss')
            plt.legend()
            plt.savefig(self._file_dir + '/effnet-loss-' + str(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.png')
            plt.show()

    def Classify_Photo_VGG19(self, photo_path, printOutput):
        # self._vgg19_model.load_weights(self._checkpoint_vgg19_path)

        img_name = photo_path
        img = image.load_img(photo_path, target_size=IMAGE_SIZE)
        # img = img.rotate(270)
        plt.imshow(img)
        plt.show()
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = self._vgg19_model.predict(images, batch_size=1)
        if printOutput:
            print("Guessed val: " + str(img_name) + " " + str(pred))
        ct = 0
        for i in pred[0]:
            ct = ct + 1
            if printOutput:
                print(str(ct) + ' - {:.2f}%'.format(i * 100))

        return pred[0]

    def Classify_Photo_EfficientNet(self, photo_path, printOutput):
        # self._efficientNet_model.load_weights(self._checkpoint_effnet_path)

        img_name = photo_path
        img = image.load_img(photo_path, target_size=IMAGE_SIZE)
        # img = img.rotate(270)
        plt.imshow(img)
        plt.show()
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = self._efficientNet_model.predict(images, batch_size=1)
        if printOutput:
            print("Guessed val: " + str(img_name) + " " + str(pred))
        ct = 0
        for i in pred[0]:
            ct = ct + 1
            if printOutput:
                print(str(ct) + ' - {:.2f}%'.format(i * 100))

        return pred[0]

    def maxelements(self, seq):
        max_indices = []
        max_val = seq[0]
        for i, val in ((i, val) for i, val in enumerate(seq) if val >= max_val):
            if val == max_val:
                max_indices.append(i + 1)
            else:
                max_val = val
                max_indices = [i + 1]

        return max_indices

    def Test_Good_Models_For_True_Accuracy(self):
        name = "goodmodels/vgg19_lr_kfoldcp1.ckpt"

        ctAll = 0
        ctCorrect = 0
        ctWrong = 0
        ctSemi = 0
        self._vgg19_model.load_weights(name)
        print("\nName: " + name + "\n")
        for img in os.listdir(self._all_photos_test_path):
            img_name = img
            img = image.load_img(self._all_photos_test_path + "/" + img, target_size=IMAGE_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            pred = self._vgg19_model.predict(images, batch_size=1)
            maxx = self.maxelements(pred[0])

            # print("Guessed val: " + str(img_name) + " " + str(maxx))
            ctAll += 1
            if int(img_name[1]) in maxx:
                if len(maxx) == 1:
                    ctCorrect += 1
                else:
                    ctSemi += 1
            else:
                ctWrong += 1

        print("Guessed: " + str(ctCorrect) + ", got wrong: " + str(ctWrong) + ", was close: " + str(
            ctSemi) + "; out of " + str(ctAll))

        name = "goodmodels/effnet_un_FT__kfoldcp4.ckpt"

        ctAll = 0
        ctCorrect = 0
        ctWrong = 0
        ctSemi = 0
        self._efficientNet_model.load_weights(name)
        print("\nName: " + name + "\n")
        for img in os.listdir(self._all_photos_test_path):
            img_name = img
            img = image.load_img(self._all_photos_test_path + "/" + img, target_size=IMAGE_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            pred = self._efficientNet_model.predict(images, batch_size=1)
            maxx = self.maxelements(pred[0])

            # print("Guessed val: " + str(img_name) + " " + str(maxx))
            ctAll += 1
            if int(img_name[1]) in maxx:
                if len(maxx) == 1:
                    ctCorrect += 1
                else:
                    ctSemi += 1
            else:
                ctWrong += 1

        print("Guessed: " + str(ctCorrect) + ", got wrong: " + str(ctWrong) + ", was close: " + str(
            ctSemi) + "; out of " + str(ctAll))



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
        elif mode == 6:
            ic.Load_Models()
            file = "ServerPhotos/2022-05-04_11-49-35-892.jpg"
            ic.Classify_Photo_VGG19(file, True)
            ic.Classify_Photo_EfficientNet(file, True)
        elif mode == 11:
            ic.Read_Images_Kfold()
            ic.Train_VGG19_Kfold()
        elif mode == 12:
            ic.Read_Images_Kfold()
            ic.Train_EfficientNet_Kfold()
        elif mode == 13:
            ic.Read_Images_Kfold()
            ic.Train_VGG19_Kfold()

            ic.Read_Images_Kfold()
            ic.Train_EfficientNet_Kfold()
        elif mode == 14:
            ic.Test_All_Models_For_True_Accuracy()
        elif mode == 15:
            ic.Read_Images_Kfold()
            ic.Train_EfficientNet_Kfold_FineTune()
        elif mode == 16:
            ic.Test_Good_Models_For_True_Accuracy()
        elif mode == 0:
            break

