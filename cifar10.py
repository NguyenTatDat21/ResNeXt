import numpy as np
import tensorflow as tf
from resnext import ResNeXtModel
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

print(tf.__version__)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
input_shape = x_train.shape[1:]

model = ResNeXtModel(input_shape=input_shape, nb_classes=10, structure=[3, 3, 3])
model.fit(x_train, y_train, x_test, y_test, 32, 100)

model.test(x_test, y_test)




