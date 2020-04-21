import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, GlobalAveragePooling2D, Dense, \
    Softmax, add, Lambda, concatenate
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.backend import shape
from tensorflow import Variable


class ResNeXtModel:
    def __init__(self, input_shape, nb_classes, structure, cardinality=16, width=4, l2_weight=5e-4):
        self.width = width
        self.cardinality = cardinality
        self.structure = structure
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.l2_weight = l2_weight
        self.model = self.build_resnext()
        self.model.summary()
        step = Variable(0, trainable=False)
        self.model.compile(optimizer=Adam(learning_rate=1e-3), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])

    def initial_block(self, inputs):
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer=he_normal(), kernel_regularizer=l2(self.l2_weight), use_bias=False)(
            inputs)
        x = BatchNormalization()(x)
        x = Activation(relu)(x)
        return x

    def output_block(self, inputs):
        x = GlobalAveragePooling2D()(inputs)
        x = Dense(self.nb_classes, kernel_initializer=he_normal(), kernel_regularizer=l2(self.l2_weight))(x)
        x = Softmax()(x)
        return x

    def bottleneck_block(self, inputs, nb_filters, strides, pass_filter=False):
        x_skip = inputs
        if pass_filter:
            x_skip = Conv2D(nb_filters * 2, (1, 1), padding='same', kernel_initializer=he_normal(),
                            kernel_regularizer=l2(self.l2_weight), strides=(strides, strides), use_bias=False)(x_skip)
            x_skip = BatchNormalization()(x_skip)
        x = Conv2D(nb_filters, (1, 1), padding='same', kernel_initializer=he_normal(),
                   kernel_regularizer=l2(self.l2_weight), use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation(relu)(x)

        x = self.grouped_block(x, int(nb_filters / self.cardinality), strides)

        x = Conv2D(nb_filters * 2, (1, 1), padding='same', kernel_initializer=he_normal(),
                   kernel_regularizer=l2(self.l2_weight), use_bias=False)(x)
        x = BatchNormalization()(x)

        x = add([x_skip, x])

        x = Activation(relu)(x)

        return x

    def build_resnext(self):
        inputs = Input(self.input_shape)
        x = self.initial_block(inputs)

        nb_filters = self.cardinality * self.width
        for stage, nb_block in enumerate(self.structure):
            for block_id in range(nb_block):
                if block_id == 0:
                    if stage > 0:
                        x = self.bottleneck_block(x, nb_filters, 2, pass_filter=True)
                    else:
                        x = self.bottleneck_block(x, nb_filters, 1, pass_filter=True)
                else:
                    x = self.bottleneck_block(x, nb_filters, 1)

            nb_filters *= 2

        x = self.output_block(x)
        model = Model(inputs=inputs, outputs=x)
        return model

    def fit(self, x, y, x_test, y_test, batch_size, epochs):
        self.model.fit(x=x, y=y, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=2)

    def grouped_block(self, inputs, group_size, strides):
        group_list = []

        for i in range(self.cardinality):
            x = Lambda(lambda _inputs: _inputs[:, :, :, i * group_size:(i + 1) * group_size])(inputs)
            x = Conv2D(group_size, (3, 3), padding='same', kernel_initializer=he_normal(),
                       kernel_regularizer=l2(self.l2_weight), strides=(strides, strides), use_bias=False)(x)
            group_list.append(x)

        x = concatenate(group_list)
        x = BatchNormalization()(x)
        x = Activation(relu)(x)
        return x

    def get_model(self):
        return self.model

