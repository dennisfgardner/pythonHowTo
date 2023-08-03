#!/usr/bin/env python3

"""resize.py

make the images and masks smaller
"""

import os
from pathlib import Path

import cv2
from skimage.transform import resize


if __name__ == "__main__":
    print("resizing...")

    root_path = Path("../data/NDD20/")
    # for the dolphin data set, it's either ABOVE or BELOW
    dir_name = "ABOVE"
    # the resized images and masks will be saved in their own directories
    img_save_dir = root_path/f"{dir_name}_resize"
    if not img_save_dir.exists():
        os.mkdir(img_save_dir)
    mask_save_dir = root_path/f"{dir_name}_masks_resize"
    if not mask_save_dir.exists():
        os.mkdir(mask_save_dir)

    cols = 400
    rows = 400
    for img_name in os.listdir(root_path/dir_name):
        """for each image:
            - open each image as grayscale
            - resize
            - save
        """
        print(img_name)
        img_filepath = str(root_path/dir_name/img_name)
        img = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
        img = resize(img, (rows, cols), mode="edge",
                     clip=True, preserve_range=True, anti_aliasing=True)
        cv2.imwrite(str(img_save_dir/img_name), img)

    dir_name = f"{dir_name}_masks"
    for mask_name in os.listdir(root_path/dir_name):
        """for each mask:
            - open each mask as grayscale
            - resize
            - save
        """
        print(mask_name)
        mask_filepath = str(f"{root_path/dir_name/mask_name}")
        mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        mask = resize(mask, (rows, cols), mode="edge",
                      clip=True, preserve_range=True, anti_aliasing=True)
        cv2.imwrite(str(mask_save_dir/mask_name), mask)
    print("finished")
