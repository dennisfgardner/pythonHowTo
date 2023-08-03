"""model.py

define U-net model which will segment the images using masks

code adapted from (as posted by DERRICK MWITI):
Sreenivas Bhattiprolu with the license stated as
"Feel free to copy, I appreciate if you acknowledge Python for Microscopists"


https://github.com/bnsreenu/python_for_microscopists
https://www.machinelearningnuggets.com/image-segmentation-with-u-net-define-u-net-model-from-scratch-in-keras-and-tensorflow/
https://www.kaggle.com/code/derrickmwiti/u-net-image-segmentation-model/?ref=machinelearningnuggets.com

data reference:
@inproceedings{
    Trotter2020NDD20AL,
    title={NDD20: A large-scale few-shot dolphin dataset for coarse and
        fine-grained categorisation
    },
    author={Cameron Trotter and Georgia Atkinson and Matt Sharpe and Kirsten
        Richardson and A. Stephen McGough and Nick Wright and Ben Burville and
        Per Berggren
    },
    year={2020}
}
quote from the data's website:
"
This dataset is licensed under CC-BY-NC-SA 4.0 and thus may not be used for
commercial purposes. If you wish to discuss utilising this dataset for
commercial purposes please contact the authors.

NDD20 is accompanied by the following paper: https://arxiv.org/abs/2005.13359
"
"""


import tensorflow as tf


def model(rows, cols):

    num_classes = 1

    inputs = tf.keras.layers.Input((rows, cols, 1))

    # Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(c1)
    b1 = tf.keras.layers.BatchNormalization()(c1)
    r1 = tf.keras.layers.ReLU()(b1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(c2)
    b2 = tf.keras.layers.BatchNormalization()(c2)
    r2 = tf.keras.layers.ReLU()(b2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(c3)
    b3 = tf.keras.layers.BatchNormalization()(c3)
    r3 = tf.keras.layers.ReLU()(b3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(c4)
    b4 = tf.keras.layers.BatchNormalization()(c4)
    r4 = tf.keras.layers.ReLU()(b4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(r4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(p4)
    b5 = tf.keras.layers.BatchNormalization()(c5)
    r5 = tf.keras.layers.ReLU()(b5)
    c5 = tf.keras.layers.Dropout(0.3)(r5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',
                                kernel_initializer='he_normal',
                                padding='same')(c5)

    # Expansive path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2),
                                         padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.BatchNormalization()(u6)
    u6 = tf.keras.layers.ReLU()(u6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2),
                                         padding='same')(u6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.BatchNormalization()(u7)
    u7 = tf.keras.layers.ReLU()(u7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2),
                                         padding='same')(u7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.BatchNormalization()(u8)
    u8 = tf.keras.layers.ReLU()(u8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2),
                                         padding='same')(u8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    u9 = tf.keras.layers.BatchNormalization()(u9)
    u9 = tf.keras.layers.ReLU()(u9)

    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1),
                                     activation='sigmoid')(u9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    tf.keras.utils.plot_model(model, "model.png")

    return model
