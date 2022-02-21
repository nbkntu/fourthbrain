from pycocotools.coco import COCO

class CocoUtils:
    def __init__(self,
                data_dir = './coco/annotations_trainval2017',
                data_type = 'val2017',
                class_names='./yolov3_tf2/data/coco.names'):
        ann_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
        self.coco = COCO(ann_file)

        self.class_names = [c.strip() for c in open(class_names).readlines()]

    def load_annotations(self, image_id, class_name):
        catIds = self.coco.getCatIds(catNms=[class_name]);
        ann_ids = self.coco.getAnnIds(
            imgIds=[image_id],
            catIds=catIds)
        anns = self.coco.loadAnns(ann_ids)
        return anns
