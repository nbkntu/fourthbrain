import os
import skimage.io

import mrcnn.model as modellib

from coco import coco
from mrcnn import utils



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


ROOT_DIR = os.path.abspath("./")

MODEL_DIR = os.path.join(ROOT_DIR, "model")
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

def download_model():
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def predict():
    config = InferenceConfig()
    #config.display()
    
    # create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=LOGS_DIR, config=config)

    # load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    
    # load an image from the images folder
    file_name = os.path.join(IMAGE_DIR, 'car.jpg')
    image = skimage.io.imread(file_name)
    
    # run detection
    results = model.detect([image], verbose=1)
    s = results[0]['masks'].shape
    print(results[0]['masks'].reshape((s[0], s[1])))
    print(type(results[0]['masks'].reshape((s[0], s[1]))[0,0]))
    r = results[0]  # single image
    print('classes: ', [class_names[cid] for cid in r['class_ids']])


def main():
    download_model()
    predict()


if __name__ == "__main__":
    main()
