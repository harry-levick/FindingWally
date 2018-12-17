import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import visualize as display_images
import model as modellib
from model import log
import skimage
import cv2
from numpy import array
from PIL import Image
from wally import WallyConfig


# root directory of the project
ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)
# local path to trained weights file
WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_wally_0030.h5")

config = WallyConfig(predict=True)
config.display()


# inspect the model in training or inference modes
# values: 'inference' or 'training'
TEST_MODE = "inference"

# create a model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=config.MODEL_DIR, config=config)

model.load_weights(WEIGHTS_PATH, by_name=True)
print("Loading weights ", WEIGHTS_PATH)


def color_splash(image, mask) :
    """
        Apply color splash effect.
        image: RGB image [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]
        Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # copy color pixels from the original color image where the mask is set
    if (mask.shape[-1] > 0) :
        # we're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else :
        splash = gray.astype(np.uint8)

    return splash


def where_is_wally() :
    image = skimage.io.imread(sys.argv[1])

    mask = model.detect([array(image)], verbose=1)[0]['masks']

    if (mask.shape[0] != 0) :
        image = color_splash(array(image), mask)
    else :
        print("Cant find Wally. Hmmm..")

    img = Image.fromarray(image, 'RGB')
    img.show()


if __name__ =='__main__' :
    where_is_wally()



































