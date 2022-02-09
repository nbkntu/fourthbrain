import os
import skimage.io
import numpy as np
import mask_rcnn.mrcnn.model as modellib
from matplotlib import pyplot as plt
from skimage.measure import find_contours, approximate_polygon

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

        # download model weights
        self.download_model(i_weights)

        # create model object in inference mode.
        config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.i_logs, config=config)
        # load weights trained on MS-COCO
        self.model.load_weights(self.i_weights, by_name=True)

    def download_model(self, i_weights):
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(i_weights):
            utils.download_trained_weights(i_weights)

    def predict(self,
                i_image_path: str,
                i_bounding_box: list,
                i_class_of_interest: int):

        # load an image from the images folder
        image = skimage.io.imread(i_image_path)
        bounding_box_image = image[i_bounding_box[1]:i_bounding_box[3] + 1, i_bounding_box[0]:i_bounding_box[2] + 1, :]

        # run detection
        results = self.model.detect([bounding_box_image], verbose=1)
        indexes = np.where(results[0]['class_ids'] == i_class_of_interest)
        if len(indexes) > 0:
            s = results[0]['masks'].shape
            bb_mask = results[0]['masks'][:, :, indexes[0]].reshape((s[0], s[1]))
            full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            full_mask[i_bounding_box[1]:i_bounding_box[3] + 1, i_bounding_box[0]:i_bounding_box[2] + 1] = bb_mask
            plt.imsave("./data/mask.jpg", full_mask)
            simple_mask_polygon = self.generate_contour(full_mask)

        return full_mask, simple_mask_polygon.astype(int)

    def generate_contour(self, full_mask):
        # get target mask
        m1 = full_mask
        #print(m1.shape)

        # pad 1 pixel on all sides
        m2 = np.zeros(
            (m1.shape[0] + 2, m1.shape[1] + 2), dtype=np.uint8)
        # overlay mask
        m2[1:-1, 1:-1] = m1

        # find contours, contours has x,y coordinates
        contours = find_contours(m2, level=0.5, fully_connected='low')
        # take the first one, ignore the rest
        contour = contours[0]
        # convert back to row,col coordinates, also shift back 1 pixel
        mask_polygon = np.fliplr(contour) - 1

        simple_contour = approximate_polygon(np.array(contour), tolerance=1)
        print(len(contour), len(simple_contour))

        # get the new mask polygon
        simple_mask_polygon = np.fliplr(simple_contour) - 1

        # plot the polygon
        plt.figure(figsize=(12, 12))
        plt.imshow(full_mask)
        for vertex in simple_mask_polygon:
            x, y = vertex[0], vertex[1]
            plt.scatter(x, y, color='r', s=5)

        plt.savefig("./data/contour.jpg")

        return simple_mask_polygon