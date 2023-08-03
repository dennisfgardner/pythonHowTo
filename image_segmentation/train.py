"""train.py

train a U-net to segment the dolphin fin

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

import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split

import model

print("train U-net")

root_path = Path("/home/dennis/repos/pythonHowTo/data/NDD20")

# make sure the number of images and masks are the same

img_dir_name = "ABOVE_resize"
img_names = os.listdir(root_path/img_dir_name)
num_imgs = len(img_names)

str_parts = img_dir_name.split("_")
mask_dir_name = str_parts[0] + "_masks_" + str_parts[1]
mask_names = os.listdir(root_path/mask_dir_name)
num_masks = len(mask_names)

assert num_imgs == num_masks, "number of images and masks must be the same"

# get the size of the images and masks, make sure they are the same
# assume all the images and masks have the same shape as the first

img = imread(root_path/img_dir_name/img_names[0])
mask = imread(root_path/mask_dir_name/mask_names[0])

assert img.shape == mask.shape, "image and mask shapes must be the same"

# load images and masks into memory
imgs = np.zeros((num_imgs, img.shape[0], img.shape[1], 1), dtype=np.uint8)
masks = np.zeros((num_masks, mask.shape[0], mask.shape[1], 1), dtype=bool)
for ii, img_name in enumerate(img_names):
    print(f"{ii=}")
    imgs[ii, :, :, :] = imread(root_path/img_dir_name/img_name)[:, :, np.newaxis]
    str_parts = img_name.split(".")
    mask_name = f"{str_parts[0]}_mask.{str_parts[1]}"
    temp = imread(root_path/mask_dir_name/mask_name)
    temp[temp < 200] = 0
    masks[ii, :, :, :] = temp[:, :, np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(imgs, masks,
                                                    test_size=0.33,
                                                    random_state=0)

rand_int = random.randint(0, len(X_train))
fig, axs = plt.subplots(1, 2)
axs[0].axis("off")
axs[0].imshow(X_train[rand_int])
axs[1].axis("off")
axs[1].imshow(y_train[rand_int])
plt.show()

model = model.model(img.shape[0], img.shape[1])

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        tf.keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")]

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=4,
          epochs=25, callbacks=callbacks)

loss = model.history.history['loss']
val_loss = model.history.history['val_loss']

plt.figure()
plt.plot(loss, 'r', label='Training loss')
plt.plot(val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input image', 'True mask', 'Predicted mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


i = random.randint(0, len(X_test))
sample_image = X_test[i]
sample_mask = y_test[i]
prediction = model.predict(sample_image[tf.newaxis, ...])[0]
predicted_mask = (prediction > 0.5).astype(np.uint8)
display([sample_image, sample_mask, predicted_mask])
print('finished')
