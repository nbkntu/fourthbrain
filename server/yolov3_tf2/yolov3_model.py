import cv2
import glob
import numpy as np
import os
import shutil
import tensorflow as tf
import time
import urllib

from absl import app, flags, logging
from absl.flags import FLAGS

from yolov3_tf2.yolov3_tf2.models2 import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.yolov3_tf2.dataset import transform_images
from yolov3_tf2.yolov3_tf2.utils import draw_outputs
from yolov3_tf2.yolov3_tf2.utils import load_darknet_weights

YOLOV3_COCO_MODEL_URL = 'https://pjreddie.com/media/files/yolov3.weights'


class YoloV3Model:
    def __init__(self,
                i_classes='./yolov3_tf2/data/coco.names',
                i_weights='./yolov3_tf2/model/yolov3.weights',
                i_tiny=False,
                i_num_classes=80,
                i_yolo_max_boxes=100,
                i_yolo_iou_threshold=0.5,
                i_yolo_score_threshold=0.5):

        self.download_model(i_weights)
        self.i_num_classes = i_num_classes

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)

        if i_tiny:
            self.yolo = YoloV3Tiny(i_yolo_max_boxes, i_yolo_iou_threshold, i_yolo_score_threshold, classes=i_num_classes)
        else:
            self.yolo = YoloV3(i_yolo_max_boxes, i_yolo_iou_threshold, i_yolo_score_threshold, classes=i_num_classes)

        load_darknet_weights(self.yolo, i_weights, i_tiny)
        logging.info('weights loaded')

        self.class_names = [c.strip() for c in open(i_classes).readlines()]
        logging.info('classes loaded')


    def download_model(self, i_weights):
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(i_weights):
            with urllib.request.urlopen(YOLOV3_COCO_MODEL_URL) as resp, open(i_weights, 'wb') as out:
                shutil.copyfileobj(resp, out)

    def get_class_name(self, class_name_id):
        return self.class_names[class_name_id - 1];

    def process(self,
                i_image_id: str,
                i_image_path: str,
                i_size=416,
                i_output='./data/bounding_boxes.jpg'):

        img_raw = tf.image.decode_image(
            open(i_image_path, 'rb').read(), channels=3)

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, i_size)

        t1 = time.time()
        boxes, scores, classes, nums = self.yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(self.class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), self.class_names)
        cv2.imwrite(i_output, img)
        logging.info('output saved to: {}'.format(i_output))

        return boxes, scores, classes + 1, nums, img.shape

    def non_maximum_suppression(self,
                                boxes,
                                classes):
        new_boxes = np.empty((0, 4), int)
        new_classes = np.empty((0, 1), int)
        for i in range(1, self.i_num_classes + 1):
            indices = np.argwhere(classes == i)
            if (len(indices) > 0):
                filtered_boxes = self.NMS(boxes[indices].reshape((boxes[indices].shape[0], boxes[indices].shape[2])))
                new_boxes = np.append(new_boxes, filtered_boxes, axis=0)
                new_classes = np.append(new_classes, np.full(filtered_boxes.shape[0], i))

        return new_boxes, new_classes

    def NMS(self, boxes, overlapThresh = 0.4):
        # Return an empty list, if no boxes given
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]  # x coordinate of the top-left corner
        y1 = boxes[:, 1]  # y coordinate of the top-left corner
        x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
        y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
        # Compute the area of the bounding boxes and sort the bounding
        # Boxes by the bottom-right y-coordinate of the bounding box
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
        # The indices of all boxes at start. We will redundant indices one by one.
        indices = np.arange(len(x1))
        for i,box in enumerate(boxes):
            # Create temporary indices
            temp_indices = indices[indices!=i]
            # Find out the coordinates of the intersection box
            xx1 = np.maximum(box[0], boxes[temp_indices,0])
            yy1 = np.maximum(box[1], boxes[temp_indices,1])
            xx2 = np.minimum(box[2], boxes[temp_indices,2])
            yy2 = np.minimum(box[3], boxes[temp_indices,3])
            # Find out the width and the height of the intersection box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / areas[temp_indices]
            # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index
            if np.any(overlap) > overlapThresh:
                indices = indices[indices != i]
        #return only the boxes at the remaining indices
        return boxes[indices].astype(int)