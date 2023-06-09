import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import shutil
import zipfile

# Import Mask RCNN
from config import Config
import utils
import model as modellib
import visualize
from model import log
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

# Begin helper functions for Python/C API
def LoadImage(image_path):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def LoadReadyWeights(weight_path):
    test_model, inference_config = load_inference_model(1, weight_path)
    return test_model

def edge_length(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def angle_between_three_points(pointA, pointB, pointC):
    x1x2s = math.pow((pointA[0] - pointB[0]),2)
    x1x3s = math.pow((pointA[0] - pointC[0]),2)
    x2x3s = math.pow((pointB[0] - pointC[0]),2)
    
    y1y2s = math.pow((pointA[1] - pointB[1]),2)
    y1y3s = math.pow((pointA[1] - pointC[1]),2)
    y2y3s = math.pow((pointB[1] - pointC[1]),2)

    return np.arccos((x1x2s + y1y2s + x2x3s + y2y3s - x1x3s - y1y3s)/(2*math.sqrt(x1x2s + y1y2s)*math.sqrt(x2x3s + y2y3s)))

def max_diameter(polygon):
    max_distance = 0
    for i in range(len(polygon)):
        for j in range(i+1, len(polygon)):
            distance = edge_length(polygon[i], polygon[j])
            if distance > max_distance:
                max_distance = distance
    return max_distance

def GetCornersFromGeneratedMask(image, test_model, tolerance = 10, per_corner = 21, scale_factor = (0.95, 0.95)):
    r = test_model.detect([image])[0]
    object_count = len(r["class_ids"])
    corner_list = []
    prev_corner1 = ()
    prev_corner2 = ()
    for i in range(object_count):
        mask = r["masks"][:, :, i]
        contours = visualize.get_mask_contours(mask)
        for cnt in contours:
            corners = cnt.reshape(-1, 2)
            for j in range(len(corners)):
                if (j % per_corner == 0):
                    forecolors = visualize.random_colors(80)
                    if prev_corner1 == ():
                        prev_corner1 = corners[j]
                        x, y = prev_corner1
                        corner_list.append((x, y))
                    elif prev_corner2 == ():
                        prev_corner2 = corners[j]
                        x, y = prev_corner2
                        corner_list.append((x, y))
                    else:
                        angle = math.degrees(angle_between_three_points(prev_corner1, prev_corner2, corners[j]))

                        if (angle > 180 - tolerance):
                            x2, y2 = prev_corner2
                            x1, y1 = corners[j]
                            corner_list.remove((x2, y2))
                            corner_list.append((x1, y1))
                            prev_corner2 = corners[j]
                        else:
                            x1, y1 = corners[j]
                            corner_list.append((x1, y1))
                            prev_corner1 = prev_corner2
                            prev_corner2 = corners[j]

    poly = Polygon(corner_list)
    centroid = poly.centroid
    cx, cy = map(int, (centroid.x, centroid.y))

    sel_x, sel_y = (0, 0)
    sel_dist = 0
    for i in range(len(corner_list)):
        x, y = corner_list[i]
        dist = math.sqrt((cx - x)**2 + (cy - y)**2)
        if (sel_x - cx)*(sel_y - cy) < (x - cx)*(y - cy) or sel_dist == 0 or dist < sel_dist:
            sel_dist = dist
            sel_x = x
            sel_y = y

    left_up_x = sel_x
    left_up_y = sel_y
    right_down_x = cx + cx - sel_x
    right_down_y = cy + cy - sel_y
    diff_horizontal = abs(left_up_x - right_down_x) - abs(left_up_x - right_down_x) * scale_factor[0]
    diff_vertical = abs(left_up_y - right_down_y) - abs(left_up_y - right_down_y) * scale_factor[1]

    left_up_x -= int(diff_horizontal / 2)
    left_up_y -= int(diff_vertical / 2)
    right_down_x += int(diff_horizontal / 2)
    right_down_y += int(diff_vertical / 2)

    image = cv2.rectangle(image, (left_up_x, left_up_y), (right_down_x, right_down_y), (255, 0, 0), 2)

    return ((left_up_x, left_up_y), (right_down_x, right_down_y))

def SaveImage(Image, Coords, Path):
    Image = cv2.rectangle(Image, Coords[0], Coords[1], (255, 0, 0), 2)
    cv2.imwrite(Path, Image)
    print("Saved: ", os.path.abspath(Path))
# End helper functions for Python/C API

class CustomConfig(Config):
    def __init__(self, num_classes):

        if num_classes > 1:
            raise ValueError("{} classes were found. This is only a DEMO version for evaluation purposes, and it only supports 1 class. Get the PRO version to"
                  " continue the training: https://pysource.com/mask-rcnn-training-pro/ ".format(num_classes))

        classes_number = num_classes
        super().__init__()
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes
    NUM_CLASSES = 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    ETF_C = 2

    DETECTION_MIN_CONFIDENCE = 0.9



"""
NOTEBOOK PREFERENCES
"""
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class CustomDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_custom(self, annotation_json, images_dir, dataset_type="train"):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """

        # Load json from file
        print("Annotation json path: ", annotation_json)
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()


        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']

            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}

        # Split the dataset, if train, get 90%, else 10%
        len_images = len(coco_json['images'])
        if dataset_type == "train":
            img_range = [int(len_images / 9), len_images]
        else:
            img_range = [0, int(len_images / 9)]

        for i in range(img_range[0], img_range[1]):
            image = coco_json['images'][i]
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        #print("Class_ids, ", class_ids)
        return mask, class_ids

    def count_classes(self):
        class_ids = set()
        for image_id in self.image_ids:
            image_info = self.image_info[image_id]
            annotations = image_info['annotations']

            for annotation in annotations:
                class_id = annotation['category_id']
                class_ids.add(class_id)

        class_number = len(class_ids)
        return class_number

def load_training_model(config):
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        print(COCO_MODEL_PATH)
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    return model


def display_image_samples(dataset_train):
    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)

    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

def load_image_dataset(annotation_path, dataset_path, dataset_type):
    dataset_train = CustomDataset()
    dataset_train.load_custom(annotation_path, dataset_path, dataset_type)
    dataset_train.prepare()
    return dataset_train


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
def train_head(model, dataset_train, dataset_val, config):
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=5,
            layers='heads')


""" DETECTION TEST YOUR MODEL """

class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def extract_images(my_zip, output_dir):
    # Make directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with zipfile.ZipFile(my_zip) as zip_file:
        count = 0
        for member in zip_file.namelist():
            filename = os.path.basename(member)
            # skip directories
            if not filename:
                continue
            count += 1
            # copy file (taken from zipfile's extract)
            source = zip_file.open(member)
            target = open(os.path.join(output_dir, filename), "wb")
            with source, target:
                shutil.copyfileobj(source, target)
        print("Extracted: {} images".format(count))


def load_test_model(num_classes):
    inference_config = InferenceConfig(num_classes)

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model, inference_config

def load_inference_model(num_classes, model_path):
    inference_config = InferenceConfig(num_classes)

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=model_path)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    #model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model, inference_config

def test_random_image(test_model, dataset_val, inference_config):
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    # log("image_meta", image_meta)
    # log("gt_class_id", gt_class_id)
    # log("gt_bbox", gt_bbox)
    # log("gt_mask", gt_mask)

    # Model result
    print("Trained model result")
    results = test_model.detect([original_image], verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax(), show_bbox=False)

    print("Annotation")
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_val.class_names, figsize=(8, 8))



