import numpy as np
import tensorflow as tf
from resnext import ResNeXtModel
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

print(tf.__version__)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
input_shape = x_train.shape[1:]

generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(x_train)
batch_size = 100

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=1e-6)

callbacks = [lr_reducer]
model = ResNeXtModel(input_shape=input_shape, nb_classes=10, structure=[3, 3, 3], cardinality=8, width=4).get_model()
model.fit(generator.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) // batch_size, epochs=300, callbacks=callbacks, validation_data=(x_test, y_test), validation_steps=x_test.shape[0] // batch_size, verbose=2)





