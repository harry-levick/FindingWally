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
import wally


# root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(""))
sys.path.append(ROOT_DIR)
# local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_wally_0030.h5")

# dataset
config = wally.WallyConfig()

config.display()

# device to load the neural network on
DEVICE = "/cpu:0"

# inspect the model in training or inference modes
# values: 'inference' or 'training'
TEST_MODE = "inference"

# create a model in inference mode
with tf.device(DEVICE) :
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_PATH, config=config)

model.load_weights(MODEL_PATH, by_name=True,
                   exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc",
                             "mrcnn_bbox", "mrcnn_mask"])
print("Loading weights ", MODEL_PATH)

def scale(image, max_size, method=Image.ANTIALIAS) :
    """
        resize 'image' to 'max_size' keeping the aspect ratio
        and place it in center of white 'max_size' image
    """
    image.thumbnail(max_size, method)
    offset = (int((max_size[0] - image.size[0]) / 2), int((max_size[1] - image.size[1]) / 2))
    back = Image.new("RGB", max_size, "white")
    back.paste(image, offset)

    return back

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

def draw_box(box, image_np):
    """
        draws a rectanlge on the image
    """
    #expand the box by 50%
    box += np.array([-(box[2] - box[0])/2, -(box[3] - box[1])/2, (box[2] - box[0])/2, (box[3] - box[1])/2])

    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    #draw blurred boxes around box
    ax.add_patch(patches.Rectangle((0,0),box[1]*image_np.shape[1], image_np.shape[0],linewidth=0,edgecolor='none',facecolor='w',alpha=0.8))
    ax.add_patch(patches.Rectangle((box[3]*image_np.shape[1],0),image_np.shape[1], image_np.shape[0],linewidth=0,edgecolor='none',facecolor='w',alpha=0.8))
    ax.add_patch(patches.Rectangle((box[1]*image_np.shape[1],0),(box[3]-box[1])*image_np.shape[1], box[0]*image_np.shape[0],linewidth=0,edgecolor='none',facecolor='w',alpha=0.8))
    ax.add_patch(patches.Rectangle((box[1]*image_np.shape[1],box[2]*image_np.shape[0]),(box[3]-box[1])*image_np.shape[1], image_np.shape[0],linewidth=0,edgecolor='none',facecolor='w',alpha=0.8))

    return fig, ax


def where_is_wally() :
    image = Image.open(os.path.join(ROOT_DIR, "input.jpg"))
    image = scale(image, (404, 718))

    with tf.device(DEVICE) :
        results = model.detect([array(image)], verbose=1)

    if (results[0]['masks'].shape[0] != 0) :
        print(results[0]['masks'][0])
        image = color_splash(array(image), results[0]['masks'])
        #image, boxed = draw_box(image, results[0]['masks'])

    image = array(image)
    skimage.io.imsave("output1.png", image)


if __name__ =='__main__' :
    where_is_wally()



































