import os
import skimage.io
import numpy as np
import mask_rcnn.mrcnn.model as modellib
from matplotlib import pyplot as plt

from mask_rcnn.coco import coco
from mask_rcnn.mrcnn import utils

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRCNNModel:
    def __init__(self,
                i_classes='./config/coco.names',
                i_weights='./mask_rcnn/model/mask_rcnn_coco.h5',
                i_logs='./mask_rcnn/logs/'):
                self.i_classes = i_classes
                self.i_weights = i_weights
                self.i_logs = i_logs

                self.download_model()

    def download_model(self):
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(self.i_weights):
            utils.download_trained_weights(self.i_weights)

    def predict(self,
                i_image_path: str,
                i_bounding_box: list,
                i_class_of_interest: int):
        config = InferenceConfig()
    
        # create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=self.i_logs, config=config)

        # load weights trained on MS-COCO
        model.load_weights(self.i_weights, by_name=True)
    
        # load an image from the images folder
        image = skimage.io.imread(i_image_path)
        bounding_box_image = image[i_bounding_box[1]:i_bounding_box[3] + 1, i_bounding_box[0]:i_bounding_box[2] + 1, :]

        # run detection
        results = model.detect([bounding_box_image], verbose=1)
        indexes = np.where(results[0]['class_ids'] == i_class_of_interest)
        if len(indexes) > 0:
            s = results[0]['masks'].shape
            bb_mask = results[0]['masks'][:, :, indexes[0]].reshape((s[0], s[1]))
            full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            full_mask[i_bounding_box[1]:i_bounding_box[3] + 1, i_bounding_box[0]:i_bounding_box[2] + 1] = bb_mask
            plt.imsave("./data/mask.jpg", full_mask)

        return full_mask