import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import json
from pycocotools import mask
from skimage import measure
import cv2
import imutils
import matplotlib.patches as patches
from PIL import Image



def checkFashionIsExist(fashion:list,file_name:str)->int:
    fashionClass=file_name.split("_")[0]
    if fashionClass=="bag" and 1 in fashion:
        return 1
    elif fashionClass=="footwear" and 2 in fashion:
        return 2
    elif fashionClass=="outer" and 3 in fashion:
        return 3
    elif fashionClass=="dress" and 4 in fashion:
        return 4
    elif fashionClass=="sunglasses" and 5 in fashion:
        return 5
    elif fashionClass=="pants" and 6 in fashion:
        return 6
    elif fashionClass=="top" and 7 in fashion:
        return 7
    elif fashionClass=="shorts" and 8 in fashion:
        return 8
    elif fashionClass=="skirt" and 9 in fashion:
        return 9
    elif fashionClass=="headwear" and 10 in fashion:
        return 10
    else:
        return -1


def getFashion(fashion: list, file_name: str):
    """
        self.add_class("custom", 1, "bag") 1
        self.add_class("custom", 2, "belt") -
        self.add_class("custom", 3, "boots") 2
        self.add_class("custom", 4, "footwear") 2
        self.add_class("custom", 5, "outer") 3
        self.add_class("custom", 6, "dress") 4
        self.add_class("custom", 7, "sunglasses") 5
        self.add_class("custom", 8, "pants") 6
        self.add_class("custom", 9, "top") 7
        self.add_class("custom", 10, "shorts") 8
        self.add_class("custom", 11, "skirt") 9
        self.add_class("custom", 12, "headwear") 10
        self.add_class("custom", 13, "scarf/tie") -
    """
    season = file_name.split("_")[1]
    seasonName=""
    if season == "ilkb":
        seasonName = 1
    elif season == "yaz":
        seasonName = 2
    elif season == "sonb":
        seasonName = 3
    elif season == "kis":
        seasonName = 4
    elif season == "4mev":
        seasonName = 5

    if fashion == 1:
        if seasonName == 1:
            return 1
        elif seasonName == 2:
            return 2
        elif seasonName == 3:
            return 3
        elif seasonName == 4:
            return 4
        elif seasonName == 5:
            return 5
    elif fashion in [2, 13]:
        return -1
    elif fashion in [3, 4]:
        if seasonName == 1:
            return 6
        elif seasonName == 2:
            return 7
        elif seasonName == 3:
            return 8
        elif seasonName == 4:
            return 9
        elif seasonName == 5:
            return 10
    elif fashion == 5:
        if seasonName == 1:
            return 11
        elif seasonName == 2:
            return 12
        elif seasonName == 3:
            return 13
        elif seasonName == 4:
            return 14
        elif seasonName == 5:
            return 15
    elif fashion == 6:
        if seasonName == 1:
            return 16
        elif seasonName == 2:
            return 17
        elif seasonName == 3:
            return 18
        elif seasonName == 4:
            return 19
        elif seasonName == 5:
            return 20
    elif fashion == 7:
        if seasonName == 1:
            return 21
        elif seasonName == 2:
            return 22
        elif seasonName == 3:
            return 23
        elif seasonName == 4:
            return 24
        elif seasonName == 5:
            return 25
    elif fashion == 8:
        if seasonName == 1:
            return 26
        elif seasonName == 2:
            return 27
        elif seasonName == 3:
            return 28
        elif seasonName == 4:
            return 29
        elif seasonName == 5:
            return 30
    elif fashion == 9:
        if seasonName == 1:
            return 31
        elif seasonName == 2:
            return 32
        elif seasonName == 3:
            return 33
        elif seasonName == 4:
            return 34
        elif seasonName == 5:
            return 35
    elif fashion == 10:
        if seasonName == 1:
            return 36
        elif seasonName == 2:
            return 37
        elif seasonName == 3:
            return 38
        elif seasonName == 4:
            return 39
        elif seasonName == 5:
            return 40
    elif fashion == 11:
        if seasonName == 1:
            return 41
        elif seasonName == 2:
            return 42
        elif seasonName == 3:
            return 43
        elif seasonName == 4:
            return 44
        elif seasonName == 5:
            return 45
    elif fashion == 12:
        if seasonName == 1:
            return 46
        elif seasonName == 2:
            return 47
        elif seasonName == 3:
            return 48
        elif seasonName == 4:
            return 49
        elif seasonName == 5:
            return 50


def generateAnnotation(annotations: dict, segmentation: list, fashions:list, file_name, widht, height):
    fashion=checkFashionIsExist(fashions,file_name)
    fashions=list(fashions)
    if fashion>0:
        x, y = [], []
        regions = []
        k=fashions.index(fashion)
        #for k in range(len(segmentation)):
        for i in range(0, len(segmentation[k][0]), 1):
            x.append(segmentation[k][0][i][1])
            y.append(segmentation[k][0][i][0])

        shape_attributes = {}
        shape_attributes["name"] = "polygon"
        shape_attributes["all_points_x"] = x
        shape_attributes["all_points_y"] = y

        region_attributes = {}

        fash = getFashion(fashion,file_name)
        region_attributes["fashion"] = str(fash)


        regions.append({"shape_attributes": shape_attributes, "region_attributes": region_attributes})

        filename = file_name
        size = -1

        file_attributes = {}
        file_attributes["widht"] = widht
        file_attributes["height"] = height

        filenameParent = file_name.split(".")[0]

        annotations[filenameParent] = {"filename": filename, "size": -1, "regions": regions,
                                       "file_attributes": file_attributes}
    else:
        print("Model found anything !! (pn.272) ")

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples\\custom\\"))  # To find local version
import coco

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
CUSTOM_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_Modanet.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(CUSTOM_MODEL_PATH):
    utils.download_trained_weights(CUSTOM_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
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
# model.load_weights(CUSTOM_MODEL_PATH, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')

# #Update the class names in the order mentioned in the custom.py file
class_names = ['BG', 'bag', 'belt', 'boots', 'footwear', 'outer', 'dress', 'sunglasses', 'pants', 'top', 'shorts',
               'skirt'  , 'headwear', 'scarf/tie']

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "image")

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

annotations = {}

xx=0
for file_name in file_names:
    print(xx)
    xx+=1
    try:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
        height, width, channels = image.shape
        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])


        # Crop mask
        masked = r["masks"]
        k = 0
        for i in range(masked.shape[2]):
            image = cv2.imread("../../images/" + file_name)
            height, width, channels = image.shape
            image = imutils.resize(image, width=width)
            for j in range(image.shape[2]):
                image[:, :, j] = image[:, :, j] * masked[:, :, i]

            f = file_name.split(".")[0]
            filename = "segmentation/" + f + "_segment_%d.jpg" % k
            cv2.imwrite(filename, image)
            k += 1
            


        # Generate vgg Annotation
        masked = r["masks"].transpose(2, 0, 1)

        segmented = []
        for testMask in masked:
            tempMask = np.zeros((np.size(testMask, 0), np.size(testMask, 1)), dtype=np.uint8)
            for i in range(len(testMask)):
                for j in range(len(testMask[i])):
                    if testMask[i][j] == True:
                        tempMask[i][j] = 1
                    else:
                        tempMask[i][j] = 0

            fortran_ground_truth_binary_mask = np.asfortranarray(tempMask)
            encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
            ground_truth_area = mask.area(encoded_ground_truth)
            ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
            contours = measure.find_contours(tempMask, 0.5)

            segmented.append(contours)

        generateAnnotation(annotations, segmented, r['class_ids'], file_name, width, height)
    except Exception as err:
        print("Hata var: ", file_name)
        print(err)

with open('custom_dataset_annotations.json', 'w') as outfile:
    json.dump(annotations, outfile, indent=4)
