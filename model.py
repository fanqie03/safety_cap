import argparse
import os

import tensorflow as tf

keras = tf.keras
layers = keras.layers
K = tf.keras.backend


def simple_net(input_shape=(112, 112, 3), classes_num=5):
    inputs = keras.Input(shape=input_shape)

    x = keras.Sequential([
        layers.Conv2D(32, 3),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3),
        layers.MaxPool2D(),

        layers.Conv2D(64, 3),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3),
        layers.MaxPool2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(classes_num)
    ])(inputs)

    return keras.Model(inputs=inputs, outputs=x)