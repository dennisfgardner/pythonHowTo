#!/usr/bin/env python3

"""create masks.py

I want the image segmentation algorithm to output masks.
Ones will be the dolphin fin and zeros will be everything else.
To start, I need to read the label's json and convert the polynomial bounding
box to a binary mask.

"""

import os
import json
from pathlib import Path

import cv2
import numpy as np


if __name__ == "__main__":
    print("creating masks...")

    root_path = Path("../data/NDD20/")
    # for the dolphin data set, it's either ABOVE or BELOW
    dir_name = "ABOVE"
    # the masks will be saved in their own directory
    save_dir = root_path/f"{dir_name}_masks"
    if not save_dir.exists():
        os.mkdir(save_dir)

    with open(root_path/f"{dir_name}_LABELS.json") as file:
        """for each image:
            - get the coordinates of the dolphin fin
            - use the coords to create a binary mask
            - save the mask
        """
        labels = json.load(file)
        for key in labels:
            print(f"working on {key}")
            regions = labels[key]["regions"]
            for region in regions:
                xpts = region["shape_attributes"]["all_points_x"]
                ypts = region["shape_attributes"]["all_points_y"]
                assert len(xpts) == len(ypts), "number of x and y points must be the same"
                vertices = []
                for x, y in zip(xpts, ypts):
                    vertex = [x, y]
                    vertices.append(vertex)
                vertices = np.array([vertices], dtype=np.int32)

                # make the mask
                mask = np.zeros((3456, 5184), dtype=np.uint8)
                cv2.fillPoly(mask, vertices, 255)
                save_name = f"{labels[key]['filename'][:-4]}_mask.jpg"
                print(save_name)
                cv2.imwrite(str(save_dir/save_name), mask)
