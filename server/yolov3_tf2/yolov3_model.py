import time
import cv2
import numpy as np
import tensorflow as tf

from absl import app, flags, logging
from absl.flags import FLAGS

from yolov3_tf2.yolov3_tf2.models2 import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.yolov3_tf2.dataset import transform_images
from yolov3_tf2.yolov3_tf2.utils import draw_outputs

class YoloV3Model:
    def __init__(self,
                i_classes='./yolov3_tf2/data/coco.names',
                i_weights='./yolov3_tf2/checkpoints/yolov3.tf',
                i_tiny=False,
                i_num_classes=80,
                i_yolo_max_boxes=100,
                i_yolo_iou_threshold=0.5,
                i_yolo_score_threshold=0.5):
        
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)    
        
        if i_tiny:
            self.yolo = YoloV3Tiny(i_yolo_max_boxes, i_yolo_iou_threshold, i_yolo_score_threshold, classes=i_num_classes)
        else:
            self.yolo = YoloV3(i_yolo_max_boxes, i_yolo_iou_threshold, i_yolo_score_threshold, classes=i_num_classes)

        self.yolo.load_weights(i_weights).expect_partial()
        logging.info('weights loaded')
        
        self.class_names = [c.strip() for c in open(i_classes).readlines()]
        logging.info('classes loaded')


    def process(self,
                i_image_id: str, 
                i_image_path: str,
                i_size=416,
                i_output='./yolov3_tf2/output.jpg'):

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

        return boxes, scores, classes, nums, img.shape
