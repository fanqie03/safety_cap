import argparse
import os

import tensorflow as tf

keras = tf.keras
layers = keras.layers
K = tf.keras.backend
Sequential = keras.Sequential


def simple_net(input_shape=(112, 112, 3), num_classes=5):

    x = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu',
                      input_shape=input_shape),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPool2D(),

        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu'),
        layers.MaxPool2D(),

        # layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        # layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return x


def simple_net_v2(input_shape=(112, 112, 3), num_classes=5):
    model = Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3),
                            activation='relu',
                            input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def mobilenet_v2(input_shape=(112, 112, 3), num_classes=5):
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
        classes=num_classes
    )

    x = base_model.output

    # x = layers.Flatten()(x)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dense(1024, activation='relu', use_bias=False)(x)
    x = layers.Dense(5, activation='softmax')(x)

    model = keras.Model(inputs=base_model.input, outputs=x)

    return model