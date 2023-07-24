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


print('*'*80)
print('*'*80)
print('*'*80)

# Path is used for cross platform filename compatibility
# it's assumed the each image class is in its own directory
PATH_TO_DATA = Path(r'./data/kagglecatsanddogs_5340/PetImages')
img_classes = os.listdir(PATH_TO_DATA)
print(f'The data contains the following classes: {img_classes}')


# filter out corrupt images
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

num_bad_imgs = 0
for img_class in img_classes:
    folder_path = PATH_TO_DATA/img_class
    for fname in folder_path.iterdir():
        try:
            fobj = open(fname, "rb")
            is_jfif = b'JFIF' in fobj.peek()
        finally:
            fobj.close()

        if not is_jfif:
            num_bad_imgs += 1
            print(f'removing {fname}')
            # Delete corrupted image
            os.remove(fname)

print(f'{num_bad_imgs} bad images removed')


# generate the dataset
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

image_size = (180, 180)
batch_size = 16

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_TO_DATA,
    validation_split=0.2,
    shuffle=True,
    seed=21887,
    subset="both",
    image_size=image_size,
    batch_size=batch_size,
)

# visualize some data
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

# augment the data
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.show()

augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))

# performance
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


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
keras.utils.plot_model(model, show_shapes=True)

# train
epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
