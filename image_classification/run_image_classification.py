"""image classification

This code is adapted from [fchollet's Keras documentation]\
    (https://keras.io/examples/vision/image_classification_from_scratch/).
The data was downloaded from [Kaggle]\
    (https://www.microsoft.com/en-us/download/details.aspx?id=54765).
"""

import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# from image_classification import make_model

image_size = (180, 180)


PATH_TO_DATA = Path(r'./data')


# Load model and the weights
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
model.load_weights("./save_at_25.keras")

img = keras.utils.load_img(
    PATH_TO_DATA/"IMG_3211.jpeg", target_size=image_size
)
img = keras.utils.load_img(
    PATH_TO_DATA/"IMG_4964.jpeg", target_size=image_size
)
img_array = keras.utils.img_to_array(img)

plt.figure(figsize=(10, 10))
plt.imshow(img_array/255)
plt.title("Test Image")
plt.axis("off")
plt.show()

img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")