"""predict.py

predict mask using U-net

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

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split

import model as create_model

print("predict using U-net")

image_size = (400, 400)

root_path = Path("/home/dennis/repos/pythonHowTo/data/NDD20/ABOVE_resize")


# Load model and the weights
model = create_model.model(image_size[0], image_size[1])

model.load_weights("/home/dennis/repos/pythonHowTo/image_segmentation/save_at_17.keras")

model.summary()

num = np.random.randint(0, 2201)
img = cv2.imread(str(root_path/f"{num}.jpg"), cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
img = tf.expand_dims(img, 0)  # Create batch axis

predicted_mask = model.predict(img)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches(10, 10)
axs[0].imshow(img[0, :, :, 0])
axs[0].set_title(f"Test Image {num}")
axs[0].set_axis_off()

axs[1].imshow(predicted_mask[0, :, :, 0])
axs[1].set_title(f"Predicted Mask {num}")
axs[1].set_axis_off()
plt.show()
