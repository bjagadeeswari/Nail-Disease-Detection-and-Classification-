from keras.layers import MaxPool2D
from keras.models import Sequential
import cv2 as cv
import numpy as np
from keras.layers import Conv2D, Dense
from torch.nn import Flatten

from Evaluation import evaluation


def VGG_16(weights_path=None, num_of_class=None):
    model = Sequential()
    model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=num_of_class, activation="softmax"))
    return model


def Model_VGG16(Train_Data, Train_Tar, Test_Data, Test_Tar):
    ## VGG16
    IMG_SIZE = [32, 32, 3]
    Train1 = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Train1[i, :, :] = cv.resize(Train_Data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Train = Train1.reshape(Train1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Test1 = np.zeros((Test_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Test_Data.shape[0]):
        Test1[i, :, :] = np.resize(Test_Data[i], (IMG_SIZE[0],IMG_SIZE[1] * IMG_SIZE[2] ))
    Test = Test1.reshape(Test1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    model = VGG_16(num_of_class=Train_Tar.shape[1])
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(Train, Train_Tar, epochs=5, steps_per_epoch=1)
    predict = model.predict(Test).astype('int')
    Eval = evaluation(predict, Test_Tar)
    return np.asarray(Eval).ravel()
