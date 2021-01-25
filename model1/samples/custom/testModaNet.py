import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples\\custom\\"))  # To find local version
import custom

#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs\\custom")

# Local path to trained weights file
CUSTOM_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_Modanet.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(CUSTOM_MODEL_PATH):
    utils.download_trained_weights(CUSTOM_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(custom.CustomConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(CUSTOM_MODEL_PATH, by_name=True)
#model.load_weights(CUSTOM_MODEL_PATH, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')

# #Update the class names in the order mentioned in the custom.py file
class_names=['BG','bag','belt','boots','footwear','outer','dress','sunglasses','pants','top','shorts','skirt'
             ,'headwear','scarf/tie']


# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

for file_name in file_names:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

# Run detection
    results = model.detect([image], verbose=1)

# Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
    print(r['class_ids'])
print()

